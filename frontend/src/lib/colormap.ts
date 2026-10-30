// Turbo colormap (Mikhailov, Google 2019), polynomial approximation. Perceptually smoother
// than jet, which CAM figures traditionally use, with the same blue → red reading.

function channel(x: number, c: readonly number[]): number {
  const v = c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * (c[4] + x * c[5]))));
  return Math.round(255 * Math.min(1, Math.max(0, v)));
}

const R = [0.13572138, 4.6153926, -42.66032258, 132.13108234, -152.94239396, 59.28637943];
const G = [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604];
const B = [0.1066733, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973];

/** 256 RGB entries, index = 8-bit heatmap value. */
export const TURBO: readonly [number, number, number][] = Array.from({ length: 256 }, (_, i) => {
  const x = i / 255;
  return [channel(x, R), channel(x, G), channel(x, B)];
});

export function turboGradientCss(): string {
  const stops = [0, 0.2, 0.4, 0.6, 0.8, 1].map((t) => {
    const [r, g, b] = TURBO[Math.round(t * 255)];
    return `rgb(${r} ${g} ${b}) ${t * 100}%`;
  });
  return `linear-gradient(to right, ${stops.join(", ")})`;
}
