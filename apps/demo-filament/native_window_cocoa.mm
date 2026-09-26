// macOS: the NSView Filament's Vulkan (MoltenVK) backend needs, with a CAMetalLayer sized to the backing pixels
// (the approach of Filament's own sample framework). Not used on other platforms.
#import <Cocoa/Cocoa.h>
#import <QuartzCore/QuartzCore.h>

void* prepareMetalView(void* nsWindow) {
  NSWindow* window = (__bridge NSWindow*)nsWindow;
  [window setColorSpace:[NSColorSpace sRGBColorSpace]];
  NSView* view = [window contentView];
  [view setWantsLayer:YES];
  CAMetalLayer* layer = [CAMetalLayer layer];
  layer.bounds = view.bounds;
  layer.autoresizingMask = kCALayerWidthSizable | kCALayerHeightSizable;
  layer.delegate = (id<CALayerDelegate>)view;
  layer.drawableSize = [view convertSizeToBacking:view.bounds.size];
  layer.contentsScale = view.window.backingScaleFactor;
  layer.opaque = YES;
  [view setLayer:layer];
  return (__bridge void*)layer;
}
