# Website videos

Copy representative evaluation videos into this directory using these names:

```text
teaser.mp4
official-diffusion-h12.mp4
abd-root-h12.mp4
abd-root-h15.mp4
abd-all-projected.mp4
ppo-locomotion.mp4
```

Recommended encoding: MP4, H.264, `yuv420p`, 1280x720 or smaller, no audio,
and `+faststart`. Example:

```sh
ffmpeg -i input.mp4 -vf "scale=1280:-2" -c:v libx264 -preset slow -crf 24 -pix_fmt yuv420p -movflags +faststart -an teaser.mp4
```
