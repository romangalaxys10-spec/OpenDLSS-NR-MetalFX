#!/usr/bin/env python3
# generate_video.py — synthesize a noisy low-res test clip as Y4M (yuv4mpeg2),
# the pipeline's platform-neutral video interchange format. Convert to/from
# mp4 with ffmpeg (scripts use it when present):
#   ffmpeg -i in.mp4 -pix_fmt yuv420p -f yuv4mpegpipe in.y4m
#   ffmpeg -i out.y4m -pix_fmt yuv420p out.mp4
#
# The clip: a scrolling procedural scene (ridges, water, a moving light)
# rendered at 1/2 resolution with Gaussian sensor noise + temporal flicker —
# exactly the "dirty render" the pipeline is designed to clean and upscale.
#
# Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
import argparse
import math
import struct
import sys


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def rnd(seed):
    state = seed & 0xFFFFFFFFFFFFFFFF
    while True:
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        z ^= z >> 31
        yield z


def gauss(gen):
    u1 = ((next(gen) >> 11) + 1) / 9007199254740993.0
    u2 = (next(gen) >> 11) / 9007199254740992.0
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def write_y4m(path, frames, w, h, fps_num, fps_den):
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{w} H{h} F{fps_num}:{fps_den} Ip A1:1 C420jpeg\n".encode())
        for frame in frames:
            f.write(b"FRAME\n")
            f.write(frame)


def render_frame(t, w, h, sigma, gen):
    y = bytearray(w * h)
    u = bytearray((w // 2) * (h // 2))
    v = bytearray((w // 2) * (h // 2))
    u_acc = [0.0] * ((w // 2) * (h // 2))
    v_acc = [0.0] * ((w // 2) * (h // 2))
    for py in range(h):
        for px in range(w):
            fx, fy = px + 0.5, py + 0.5
            r = g = b = 0.15 + 0.35 * (fy / h)
            r *= 0.75; g *= 0.85
            ds = math.hypot(fx - w * 0.78, fy - h * 0.2)
            if ds < w * 0.06:
                r = g = b = 1.0
            ridge1 = h * (0.62 + 0.05 * (0.6 * math.sin((fx + t * 40) / 37.0)
                                         + 0.4 * math.sin((fx + t * 40) / 13.7 + 1.7)))
            ridge2 = h * (0.78 + 0.07 * (0.6 * math.sin((fx + t * 25) / 53.0)
                                         + 0.4 * math.sin((fx + t * 25) / 19.6 + 1.7)))
            if fy > ridge2:
                r, g, b = 0.16, 0.18, 0.22
            if fy > ridge1:
                ripple = math.sin(fx * 0.35 + math.sin(fy * 0.9 + t * 2.0) * 2.0) * 0.5 + 0.5
                r, g, b = 0.05 + 0.03 * ripple, 0.07 + 0.04 * ripple, 0.10 + 0.05 * ripple
            n = gauss(gen) * sigma
            r, g, b = (clamp(r + n, 0, 1), clamp(g + n, 0, 1), clamp(b + n, 0, 1))
            yy = 16 + 219 * (0.2126 * r + 0.7152 * g + 0.0722 * b)
            uu = 128 + 224 * (-0.2126 * r - 0.7152 * g + 0.9278 * b) * 0.564
            vv = 128 + 224 * (0.9278 * r - 0.7152 * g - 0.2126 * b) * 0.713
            y[py * w + px] = int(clamp(yy, 16, 235) + 0.5)
            u_acc[(py // 2) * (w // 2) + (px // 2)] += clamp(uu, 16, 240)
            v_acc[(py // 2) * (w // 2) + (px // 2)] += clamp(vv, 16, 240)
    for i in range(len(u)):
        u[i] = int(u_acc[i] * 0.25 + 0.5)
        v[i] = int(v_acc[i] * 0.25 + 0.5)
    return bytes(y) + bytes(u) + bytes(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="noise_input.y4m")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=180)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--sigma", type=float, default=0.07, help="noise sigma 0..1")
    ap.add_argument("--seed", type=int, default=0xC0FFEE)
    args = ap.parse_args()

    gen = rnd(args.seed)
    frames = []
    for i in range(args.frames):
        frames.append(render_frame(i / args.fps, args.width, args.height, args.sigma, gen))
        if (i + 1) % 8 == 0:
            print(f"  rendered {i+1}/{args.frames}", file=sys.stderr)
    write_y4m(args.out, frames, args.width, args.height, args.fps, 1)
    print(f"wrote {args.out}: {args.frames} frames of {args.width}x{args.height} @ {args.fps} fps")


if __name__ == "__main__":
    main()
