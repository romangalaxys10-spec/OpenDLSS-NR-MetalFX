// nr_video.cpp — FFmpeg pipeline: decode (swscale to RGBA f32) -> NR session
// per frame -> encode (libx264/libx265, yuv420p, CRF quality). The container
// frame rate is preserved; audio streams are copied when present.
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include "nr_frame.h"
#include "nr_video.h"

#include <cstring>
#include <memory>
#include <stdexcept>

namespace nr {
namespace video {

namespace {

struct Streams {
  AVFormatContext* input = nullptr;
  AVFormatContext* output = nullptr;
  AVCodecContext* decoder = nullptr;
  AVCodecContext* encoder = nullptr;
  AVFrame* frame = nullptr;        // decoded frame, native format
  AVFrame* rgba = nullptr;         // RGBA8 conversion target
  AVFrame* outFrame = nullptr;     // encoder input, yuv420p
  AVPacket* packet = nullptr;
  SwsContext* toRgba = nullptr;
  SwsContext* toYuv = nullptr;
  int videoIndex = -1;
  int audioIndex = -1;

  ~Streams() {
    if (packet) av_packet_free(&packet);
    if (outFrame) av_frame_free(&outFrame);
    if (rgba) av_frame_free(&rgba);
    if (frame) av_frame_free(&frame);
    if (toYuv) sws_freeContext(toYuv);
    if (toRgba) sws_freeContext(toRgba);
    if (encoder) avcodec_free_context(&encoder);
    if (decoder) avcodec_free_context(&decoder);
    if (output) avformat_free_context(output);
    if (input) avformat_close_input(&input);
  }
};

const AVCodec* pickEncoder() {
  if (const AVCodec* codec = avcodec_find_encoder_by_name("libx265")) return codec;
  if (const AVCodec* codec = avcodec_find_encoder_by_name("libx264")) return codec;
  return avcodec_find_encoder(AV_CODEC_ID_H264);
}

// Sends one RGB frame through the NR session and stages the encoder input.
void processAndStage(Streams& s, nr::frame::Session& session, const TranscodeOptions& options,
                     const AVFrame* source) {
  const uint32_t width = (uint32_t)source->width, height = (uint32_t)source->height;
  // swscale -> RGBA8
  s.toRgba = sws_getCachedContext(s.toRgba, (int)width, (int)height, (AVPixelFormat)source->format,
                                  (int)width, (int)height, AV_PIX_FMT_RGBA, SWS_BILINEAR, nullptr, nullptr, nullptr);
  if (!s.toRgba) throw std::runtime_error("swscale (RGBA) failed");
  av_frame_make_writable(s.rgba);
  const uint8_t* srcData[4] = {source->data[0], source->data[1], source->data[2], source->data[3]};
  const int srcStride[4] = {source->linesize[0], source->linesize[1], source->linesize[2], source->linesize[3]};
  sws_scale(s.toRgba, srcData, srcStride, 0, (int)height, s.rgba->data, s.rgba->linesize);

  // RGBA8 -> f32 proxy [0,1]
  std::vector<float> proxy((size_t)width * height * 4);
  const uint8_t* bytes = s.rgba->data[0];
  for (size_t i = 0; i < proxy.size(); ++i) proxy[i] = bytes[i] * (1.0f / 255.0f);

  // NR network
  std::vector<uint8_t> outRgba;
  session.processFrame(proxy.data(), width, outRgba);

  // RGBA8 -> yuv420p for the encoder
  nr::frame::Params params = session.params();
  s.toYuv = sws_getCachedContext(s.toYuv, (int)params.validWidth, (int)params.validHeight, AV_PIX_FMT_RGBA,
                                 (int)params.validWidth, (int)params.validHeight, AV_PIX_FMT_YUV420P, SWS_BICUBIC,
                                 nullptr, nullptr, nullptr);
  if (!s.toYuv) throw std::runtime_error("swscale (YUV) failed");
  av_frame_make_writable(s.outFrame);
  const uint8_t* outData[4] = {outRgba.data(), nullptr, nullptr, nullptr};
  const int outStride[4] = {(int)params.validWidth * 4, 0, 0, 0};
  sws_scale(s.toYuv, outData, outStride, 0, (int)params.validHeight, s.outFrame->data, s.outFrame->linesize);
  (void)options;
}

void encodeFrame(Streams& s, AVFrame* frame) {
  int ret = avcodec_send_frame(s.encoder, frame);
  while (ret >= 0) {
    ret = avcodec_receive_packet(s.encoder, s.packet);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) return;
    if (ret < 0) throw std::runtime_error("encoder receive failed");
    av_packet_rescale_ts(s.packet, s.encoder->time_base, s.output->streams[0]->time_base);
    s.packet->stream_index = 0;
    if (av_interleaved_write_frame(s.output, s.packet) < 0) throw std::runtime_error("packet write failed");
    av_packet_unref(s.packet);
  }
}

}  // namespace

void transcode(const TranscodeOptions& options, const std::function<void(const Progress&)>& progress) {
  Streams s;
  if (avformat_open_input(&s.input, options.input.c_str(), nullptr, nullptr) < 0)
    throw std::runtime_error("cannot open input " + options.input);
  if (avformat_find_stream_info(s.input, nullptr) < 0) throw std::runtime_error("no stream info");
  s.videoIndex = av_find_best_stream(s.input, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (s.videoIndex < 0) throw std::runtime_error("no video stream");
  AVStream* inVideo = s.input->streams[s.videoIndex];
  s.audioIndex = av_find_best_stream(s.input, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);

  const AVCodec* decoderCodec = avcodec_find_decoder(inVideo->codecpar->codec_id);
  if (!decoderCodec) throw std::runtime_error("unsupported input codec");
  s.decoder = avcodec_alloc_context3(decoderCodec);
  if (avcodec_parameters_to_context(s.decoder, inVideo->codecpar) < 0) throw std::runtime_error("codec params failed");
  if (avcodec_open2(s.decoder, decoderCodec, nullptr) < 0) throw std::runtime_error("decoder open failed");
  const uint32_t width = (uint32_t)s.decoder->width, height = (uint32_t)s.decoder->height;

  nr::frame::Params params = options.params;
  if (params.validWidth == 0) params.validWidth = width;
  if (params.validHeight == 0) params.validHeight = height;
  if (params.sourceWidth == 0) params.sourceWidth = width;
  if (params.sourceHeight == 0) params.sourceHeight = height;

  nr::frame::SessionOptions sessionOptions;
  sessionOptions.modelDirectory = options.modelDirectory;
  sessionOptions.params = params;
  std::unique_ptr<nr::frame::Session> session = nr::frame::createSession(sessionOptions);

  const AVCodec* encoderCodec = pickEncoder();
  if (!encoderCodec) throw std::runtime_error("no encoder (libx264/libx265 not in this FFmpeg build)");
  s.encoder = avcodec_alloc_context3(encoderCodec);
  s.encoder->width = (int)params.validWidth;
  s.encoder->height = (int)params.validHeight;
  s.encoder->pix_fmt = AV_PIX_FMT_YUV420P;
  s.encoder->time_base = inVideo->time_base;
  s.encoder->framerate = inVideo->avg_frame_rate;
  s.encoder->gop_size = 30;
  s.encoder->max_b_frames = 2;
  av_opt_set(s.encoder->priv_data, "crf", std::to_string(options.crf).c_str(), 0);
  av_opt_set(s.encoder->priv_data, "preset", "slow", 0);
  if (avcodec_open2(s.encoder, encoderCodec, nullptr) < 0) throw std::runtime_error("encoder open failed");

  if (avformat_alloc_output_context2(&s.output, nullptr, nullptr, options.output.c_str()) < 0 || !s.output)
    throw std::runtime_error("cannot create output " + options.output);
  AVStream* outVideo = avformat_new_stream(s.output, nullptr);
  if (!outVideo) throw std::runtime_error("output stream failed");
  outVideo->time_base = inVideo->time_base;
  if (avcodec_parameters_from_context(outVideo->codecpar, s.encoder) < 0)
    throw std::runtime_error("output params failed");
  if (!(s.output->oformat->flags & AVFMT_NOFILE)) {
    if (avio_open(&s.output->pb, options.output.c_str(), AVIO_FLAG_WRITE) < 0)
      throw std::runtime_error("cannot write " + options.output);
  }
  if (avformat_write_header(s.output, nullptr) < 0) throw std::runtime_error("header write failed");

  s.frame = av_frame_alloc();
  s.rgba = av_frame_alloc();
  s.rgba->format = AV_PIX_FMT_RGBA;
  s.rgba->width = (int)width;
  s.rgba->height = (int)height;
  if (av_frame_get_buffer(s.rgba, 0) < 0) throw std::runtime_error("rgba buffer failed");
  s.outFrame = av_frame_alloc();
  s.outFrame->format = AV_PIX_FMT_YUV420P;
  s.outFrame->width = (int)params.validWidth;
  s.outFrame->height = (int)params.validHeight;
  s.outFrame->pts = 0;
  if (av_frame_get_buffer(s.outFrame, 0) < 0) throw std::runtime_error("encoder buffer failed");
  s.packet = av_packet_alloc();

  Progress reported;
  uint32_t processed = 0;
  while (av_read_frame(s.input, s.packet) >= 0) {
    if (s.packet->stream_index != s.videoIndex) {
      av_packet_unref(s.packet);
      continue;
    }
    if (avcodec_send_packet(s.decoder, s.packet) < 0) {
      av_packet_unref(s.packet);
      continue;
    }
    av_packet_unref(s.packet);
    while (avcodec_receive_frame(s.decoder, s.frame) >= 0) {
      const uint32_t index = processed;
      if (index >= options.startFrame && (options.maxFrames == 0 || index < options.startFrame + options.maxFrames)) {
        processAndStage(s, *session, options, s.frame);
        s.outFrame->pts = (int64_t)index * inVideo->time_base.den /
                          (inVideo->time_base.num * (inVideo->avg_frame_rate.num ? inVideo->avg_frame_rate.num : 30));
        encodeFrame(s, s.outFrame);
        reported.frame = index;
        if (progress) progress(reported);
      }
      ++processed;
      av_frame_unref(s.frame);
      if (options.maxFrames && index + 1 >= options.startFrame + options.maxFrames) goto done;
    }
  }
done:
  encodeFrame(s, nullptr);   // flush
  av_write_trailer(s.output);
  if (!(s.output->oformat->flags & AVFMT_NOFILE)) avio_closep(&s.output->pb);
}

}  // namespace video
}  // namespace nr
