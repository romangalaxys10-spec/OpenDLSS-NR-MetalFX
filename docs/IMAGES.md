# Images

`opendlss image in.png out.png [--style N] [--grain N] [--tonemap F] [--structure F] [--skin F]
[--automask]`. Sources: PNG/JPEG/BMP/TGA (stb), normalized to [0,1] display codes; output: PNG or
JPEG (quality 95). The network's RGB output is a residual, composed against the centred proxy:

```
out = clamp((head/32 + centred)*8 + 0.5, 0, 1)     centred = roundF16(roundF16(roundF16(px)-0.5)*0.125)
```

Conditioning knobs (docs/original-design/network.md has the network's own description):
`--style` (0..128), `--tonemap` local tone, `--structure` local structure, `--skin` skin structure
(< 0 auto), `--automask` the auto skin-mask lane pair, `--grain` the noise seed (different seeds,
different generated detail — batch a few and pick).
