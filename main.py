import math
import random
import pygame
import sys

# ── Display (logical window smaller; supersampled for sharpness) ───────────────
WIN_W, WIN_H = 1440, 810
RENDER_SCALE = 4
FPS = 30

# ── Plot ───────────────────────────────────────────────────────────────────────
PANEL_W = 420
X_MIN, X_MAX = -10, 10
Y_MIN, Y_MAX = -11, 11
PLOT_LEFT = PANEL_W
PLOT_RIGHT = WIN_W - 24
PLOT_TOP, PLOT_BOTTOM = 12, WIN_H - 64
ORIGIN_X = PLOT_LEFT + (PLOT_RIGHT - PLOT_LEFT) // 2
ORIGIN_Y = PLOT_TOP + (PLOT_BOTTOM - PLOT_TOP) // 2

SCALE_X = (PLOT_RIGHT - PLOT_LEFT) / (X_MAX - X_MIN)
SCALE_Y = (PLOT_BOTTOM - PLOT_TOP) / (Y_MAX - Y_MIN)
PLOT_WIDTH = PLOT_RIGHT - PLOT_LEFT

# ── Colors ─────────────────────────────────────────────────────────────────────
BG = (18, 18, 28)
GRID = (40, 40, 55)
AXIS = (90, 90, 110)
BLUE = (80, 160, 255)
RED = (255, 90, 90)
WHITE = (230, 230, 240)
GRAY = (140, 140, 160)
YELLOW = (255, 220, 60)
HIGHLIGHT = (255, 200, 50)
GLOW = (255, 180, 40)
PANEL_BG = (28, 28, 42)
BTN_BG = (50, 50, 70)
BTN_HOVER = (70, 70, 95)
CHECK_ON = (100, 200, 140)
CHECK_OFF = (60, 60, 80)
TRAIL_ALPHA = 40

# ── Layout (logical px) ────────────────────────────────────────────────────────
MARGIN = 16
SLIDER_LEFT = MARGIN
SLIDER_LABEL_W = 36
SLIDER_WIDTH = PANEL_W - MARGIN * 2 - SLIDER_LABEL_W - 52
SLIDER_HEIGHT = 12
BTN_SIZE = 28
CHECKBOX_SIZE = 16
CHECKBOX_H = 22
BOTTOM_RESERVE = 200

# ── Core state ─────────────────────────────────────────────────────────────────
coeffs = [1, 5, -2, 3]
playing = False
selected_m = 1
show_v0 = False
term_colors: dict[int, tuple[int, int, int]] = {}
phase_offsets: dict[int, float] = {}
speed_omegas: dict[int, float] = {}
v_op_colors: dict[int, tuple[int, int, int]] = {}
trail_history: list[list[list[tuple[int, int]]]] = []
MAX_TRAIL = 500
SAMPLE_MULT = 4


def s(v: float | int) -> int:
    return int(v * RENDER_SCALE)


MAX_SCREEN_DY = s(int((PLOT_BOTTOM - PLOT_TOP) * 0.4))


def max_operator_m() -> int:
    return max(0, len(coeffs) - 1)


def min_operator_m() -> int:
    return 0 if show_v0 else 1


def clamp_m(m: int) -> int:
    return max(min_operator_m(), min(m, max_operator_m()))


def random_term_color() -> tuple[int, int, int]:
    return tuple(random.randint(80, 255) for _ in range(3))


def v_operator_color(m: int) -> tuple[int, int, int]:
    if m not in v_op_colors:
        hue = (m * 47 + 15) % 360
        c = pygame.Color(0)
        c.hsla = (hue, 72, 58, 100)
        v_op_colors[m] = (max(c.r, 80), max(c.g, 80), max(c.b, 80))
    return v_op_colors[m]


def random_speed() -> float:
    """Normal-distributed angular speed, clamped positive."""
    return max(0.08, random.gauss(0.55, 0.18))


def ensure_term_meta(degree: int) -> None:
    if degree not in term_colors:
        term_colors[degree] = random_term_color()
        phase_offsets[degree] = random.uniform(0, 2 * math.pi)
        speed_omegas[degree] = random_speed()


def remove_term_meta(degree: int) -> None:
    term_colors.pop(degree, None)
    phase_offsets.pop(degree, None)
    speed_omegas.pop(degree, None)


def init_colors() -> None:
    for k in range(len(coeffs)):
        ensure_term_meta(k)


def slider_gap() -> int:
    n = max(len(coeffs), 1)
    start = slider_top_y()
    avail = WIN_H - BOTTOM_RESERVE - start
    return max(26, min(34, avail // n))


def slider_top_y() -> int:
    return 168


# ── Math ───────────────────────────────────────────────────────────────────────
def poly(x: float, c: list[float] | None = None) -> float:
    c = coeffs if c is None else c
    return sum(a * x**k for k, a in enumerate(c))


def poly_deriv(x: float, c: list[float] | None = None) -> float:
    c = coeffs if c is None else c
    return sum(k * a * x ** (k - 1) for k, a in enumerate(c) if k > 0)


def v_coeff(k: int, a: float, m: int) -> float:
    if m == 0:
        return 0.0
    return (1 - k / m) * a


def v_poly(x: float, c: list[float] | None = None, m: int | None = None) -> float:
    c = coeffs if c is None else c
    m = selected_m if m is None else m
    if m == 0:
        return float("nan")
    return sum(v_coeff(k, a, m) * x**k for k, a in enumerate(c))


def v0_stationary_xs() -> list[float]:
    """x-coordinates of P's stationary points; V₀ draws x = x* for each."""
    roots = find_stationary_points()
    deduped: list[float] = []
    for r in roots:
        if not deduped or abs(r - deduped[-1]) > 0.08:
            deduped.append(r)
    return deduped


def monomial(x: float, k: int, a: float) -> float:
    return a * x**k


def v_monomial(x: float, k: int, a: float, m: int) -> float:
    return v_coeff(k, a, m) * x**k


def world_to_screen(x: float, y: float) -> tuple[int, int]:
    sx = s(ORIGIN_X + x * SCALE_X)
    sy = s(ORIGIN_Y - y * SCALE_Y)
    return sx, sy


def find_stationary_points() -> list[float]:
    roots: list[float] = []
    step = 0.05
    x = X_MIN
    prev = poly_deriv(x)
    while x < X_MAX:
        x_next = x + step
        curr = poly_deriv(x_next)
        if prev == 0:
            roots.append(x)
        elif curr == 0:
            roots.append(x_next)
        elif prev * curr < 0:
            lo, hi = x, x_next
            for _ in range(50):
                mid = (lo + hi) / 2
                if poly_deriv(lo) * poly_deriv(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            roots.append((lo + hi) / 2)
        prev = curr
        x = x_next
    return roots


def v_matrix_entries(m: int) -> list[str]:
    n = len(coeffs)
    if m == 0:
        return [f"x={xr:.3g}" for xr in v0_stationary_xs()] or ["(no roots)"]
    return [f"{1 - k / m:.3g}" for k in range(n)]


def polynomial_segments(highlight_k: int | None = None) -> list[tuple[str, tuple[int, int, int]]]:
    segments: list[tuple[str, tuple[int, int, int]]] = [("P(x)=", WHITE)]
    first = True
    for k in range(len(coeffs) - 1, -1, -1):
        a = coeffs[k]
        if abs(a) < 1e-9:
            continue
        sign = "+" if a >= 0 else "-"
        aval = abs(a)
        coeff_str = str(int(aval)) if aval == int(aval) else f"{aval:.2g}"
        if k == 0:
            term = coeff_str
        elif k == 1:
            term = f"{coeff_str}x" if coeff_str != "1" else "x"
        else:
            sup = str(k).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
            term = f"{coeff_str}x{sup}" if coeff_str != "1" else f"x{sup}"
        if first:
            text = f"-{term}" if sign == "-" else term
            first = False
        else:
            text = f" {sign} {term}"
        color = HIGHLIGHT if k == highlight_k else WHITE
        segments.append((text, color))
    if first:
        segments.append(("0", WHITE))
    return segments


def wrap_segments(segments: list[tuple[str, tuple[int, int, int]]],
                  font: pygame.font.Font, max_w: int) -> list[list[tuple[str, tuple[int, int, int]]]]:
    lines: list[list[tuple[str, tuple[int, int, int]]]] = [[]]
    line_w = 0
    for text, color in segments:
        w = font.size(text)[0]
        if line_w + w > max_w and lines[-1]:
            lines.append([])
            line_w = 0
        lines[-1].append((text, color))
        line_w += w
    return lines


# ── UI classes ─────────────────────────────────────────────────────────────────
class Slider:
    def __init__(self, degree: int, value: float = 0.0):
        self.degree = degree
        self.value = value
        self.dragging = False
        self._update_rect()

    def _update_rect(self) -> None:
        gap = slider_gap()
        y = slider_top_y() + self.degree * gap
        track_x = SLIDER_LEFT + SLIDER_LABEL_W
        self.track_rect = pygame.Rect(s(track_x), s(y), s(SLIDER_WIDTH), s(SLIDER_HEIGHT))
        self.rect = pygame.Rect(s(SLIDER_LEFT), s(y - 4), s(PANEL_W - MARGIN * 2), s(SLIDER_HEIGHT + 10))

    def value_to_x(self) -> int:
        t = (self.value + 10) / 20
        return int(self.track_rect.left + t * self.track_rect.width)

    def x_to_value(self, mx: int) -> float:
        t = (mx - self.track_rect.left) / self.track_rect.width
        return max(-10, min(10, t * 20 - 10))

    def handle_event(self, event: pygame.event.Event, scale: int) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = (event.pos[0] * scale, event.pos[1] * scale)
            if self.rect.collidepoint(pos):
                self.dragging = True
                self.value = self.x_to_value(pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
                return True
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            pos = (event.pos[0] * scale, event.pos[1] * scale)
            self.value = self.x_to_value(pos[0])
            return True
        return False

    def draw(self, screen: pygame.Surface, font: pygame.font.Font, highlighted: bool) -> None:
        color = term_colors.get(self.degree, WHITE)
        swatch = pygame.Rect(s(SLIDER_LEFT), self.track_rect.centery - s(7), s(14), s(14))
        if highlighted:
            glow = swatch.inflate(s(6), s(6))
            glow_surf = pygame.Surface(glow.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*HIGHLIGHT, 100), glow_surf.get_rect(), border_radius=s(3))
            screen.blit(glow_surf, glow.topleft)
        pygame.draw.rect(screen, color, swatch, border_radius=s(2))

        label_text = f"a{self.degree}" + (" ←elim" if highlighted else "")
        label = font.render(label_text, True, HIGHLIGHT if highlighted else WHITE)
        screen.blit(label, (swatch.right + s(4), self.track_rect.centery - label.get_height() // 2))

        pygame.draw.rect(screen, (35, 35, 50), self.track_rect, border_radius=s(4))
        pygame.draw.rect(screen, GRAY, self.track_rect, 1, border_radius=s(4))

        if highlighted:
            glow = self.track_rect.inflate(s(8), s(8))
            glow_surf = pygame.Surface(glow.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*GLOW, 80), glow_surf.get_rect(), border_radius=s(6))
            screen.blit(glow_surf, glow.topleft)

        knob_x = self.value_to_x()
        pygame.draw.circle(screen, color, (knob_x, self.track_rect.centery), s(7))
        if highlighted:
            pygame.draw.circle(screen, HIGHLIGHT, (knob_x, self.track_rect.centery), s(9), s(2))

        val_text = font.render(f"{self.value:.1f}", True, GRAY)
        screen.blit(val_text, (self.track_rect.right + s(8), self.track_rect.centery - val_text.get_height() // 2))

class Button:
    def __init__(self, rect: pygame.Rect, label: str):
        self.rect = rect
        self.label = label
        self.hover = False

    def _pos(self, event: pygame.event.Event, scale: int) -> tuple[int, int]:
        return event.pos[0] * scale, event.pos[1] * scale

    def handle_event(self, event: pygame.event.Event, scale: int) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(self._pos(event, scale))
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(self._pos(event, scale)):
                return True
        return False

    def draw(self, screen: pygame.Surface, font: pygame.font.Font, active: bool = False) -> None:
        color = CHECK_ON if active else (BTN_HOVER if self.hover else BTN_BG)
        pygame.draw.rect(screen, color, self.rect, border_radius=s(6))
        pygame.draw.rect(screen, GRAY, self.rect, 1, border_radius=s(6))
        text = font.render(self.label, True, WHITE)
        screen.blit(text, text.get_rect(center=self.rect.center))


class Checkbox:
    def __init__(self, rect: pygame.Rect, label: str, checked: bool = False):
        self.rect = rect
        self.label = label
        self.checked = checked

    def handle_event(self, event: pygame.event.Event, scale: int) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = (event.pos[0] * scale, event.pos[1] * scale)
            box = pygame.Rect(self.rect.left, self.rect.centery - s(CHECKBOX_SIZE) // 2,
                              s(CHECKBOX_SIZE), s(CHECKBOX_SIZE))
            if box.collidepoint(pos) or self.rect.collidepoint(pos):
                self.checked = not self.checked
                return True
        return False

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        box = pygame.Rect(self.rect.left, self.rect.centery - s(CHECKBOX_SIZE) // 2,
                          s(CHECKBOX_SIZE), s(CHECKBOX_SIZE))
        pygame.draw.rect(screen, CHECK_ON if self.checked else CHECK_OFF, box, border_radius=s(3))
        pygame.draw.rect(screen, GRAY, box, 1, border_radius=s(3))
        if self.checked:
            cx, cy = box.center
            pygame.draw.line(screen, WHITE, (cx - s(4), cy), (cx - s(1), cy + s(4)), s(2))
            pygame.draw.line(screen, WHITE, (cx - s(1), cy + s(4)), (cx + s(5), cy - s(4)), s(2))
        text = font.render(self.label, True, WHITE)
        screen.blit(text, (box.right + s(8), self.rect.centery - text.get_height() // 2))


# ── Drawing helpers ────────────────────────────────────────────────────────────
def _interp_boundary(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float] | None:
    if not (math.isfinite(y0) and math.isfinite(y1)) or y0 == y1:
        return None
    for yb in (Y_MAX, Y_MIN):
        if (y0 - yb) * (y1 - yb) < 0:
            t = (yb - y0) / (y1 - y0)
            return x0 + t * (x1 - x0), yb
    return None


def _safe_eval(fn, x: float) -> float:
    try:
        y = fn(x)
        return y if math.isfinite(y) else float("nan")
    except (OverflowError, ValueError):
        return float("nan")


def sample_curve_segments(fn, num_points: int | None = None) -> list[list[tuple[int, int]]]:
    """Sample fn into screen segments, clipping at y-bounds to avoid spike artifacts."""
    n = num_points or max(PLOT_WIDTH * SAMPLE_MULT, 4000)
    segments: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    prev_x = prev_y = None
    prev_vis = False

    for i in range(n):
        x = X_MIN + i / max(n - 1, 1) * (X_MAX - X_MIN)
        y = _safe_eval(fn, x)
        vis = Y_MIN <= y <= Y_MAX

        if prev_x is not None and math.isfinite(prev_y):
            if prev_vis and not vis:
                hit = _interp_boundary(prev_x, prev_y, x, y)
                if hit:
                    current.append(world_to_screen(*hit))
                if len(current) >= 2:
                    segments.append(current)
                current = []
            elif not prev_vis and vis:
                hit = _interp_boundary(prev_x, prev_y, x, y)
                if hit:
                    current.append(world_to_screen(*hit))

        if vis:
            pt = world_to_screen(x, y)
            if current:
                dy = abs(pt[1] - current[-1][1])
                if dy > MAX_SCREEN_DY:
                    if len(current) >= 2:
                        segments.append(current)
                    current = []
            current.append(pt)
        elif current:
            if len(current) >= 2:
                segments.append(current)
            current = []

        prev_x, prev_y, prev_vis = x, y, vis

    if len(current) >= 2:
        segments.append(current)
    return segments


def draw_curve_segments(screen: pygame.Surface, segments: list[list[tuple[int, int]]],
                        color: tuple[int, int, int], width: int) -> None:
    for seg in segments:
        if len(seg) >= 2:
            pygame.draw.lines(screen, color, False, seg, width)


def draw_grid(screen: pygame.Surface) -> None:
    for x in range(int(X_MIN), int(X_MAX) + 1):
        p1 = world_to_screen(x, Y_MIN)
        p2 = world_to_screen(x, Y_MAX)
        pygame.draw.line(screen, AXIS if x == 0 else GRID, p1, p2, s(1))
    for y in range(int(Y_MIN), int(Y_MAX) + 1):
        p1 = world_to_screen(X_MIN, y)
        p2 = world_to_screen(X_MAX, y)
        pygame.draw.line(screen, AXIS if y == 0 else GRID, p1, p2, s(1))


def draw_formula_panel(screen: pygame.Surface, title_font: pygame.font.Font,
                       small_font: pygame.font.Font) -> int:
    """Draw header block; return y below it."""
    highlight_k = selected_m if selected_m > 0 and selected_m <= len(coeffs) - 1 else None
    max_w = s(PANEL_W - MARGIN * 2)
    y = s(MARGIN)
    for line in wrap_segments(polynomial_segments(highlight_k), title_font, max_w):
        x_off = s(SLIDER_LEFT)
        for text, color in line:
            surf = title_font.render(text, True, color)
            screen.blit(surf, (x_off, y))
            x_off += surf.get_width()
        y += title_font.get_linesize()

    y += s(6)
    op_text = (f"Current Operator: V{selected_m}   "
               f"(←→ m, ↑↓ terms, max V{max_operator_m()})")
    screen.blit(small_font.render(op_text, True, HIGHLIGHT), (s(SLIDER_LEFT), y))
    y += small_font.get_linesize() + s(4)

    if selected_m == 0:
        xs = v0_stationary_xs()
        if xs:
            parts = ", ".join(f"x={xr:.3g}" for xr in xs[:4])
            if len(xs) > 4:
                parts += ", …"
            note = f"V₀ = vertical lines  {{ {parts} }}"
        else:
            note = "V₀ = vertical lines  (no stationary points in range)"
        screen.blit(small_font.render(note, True, HIGHLIGHT), (s(SLIDER_LEFT), y))
        y += small_font.get_linesize()
    elif highlight_k is not None and highlight_k < len(coeffs):
        elim = small_font.render(f"a{highlight_k} is eliminated by V{selected_m}", True, HIGHLIGHT)
        screen.blit(elim, (s(SLIDER_LEFT), y))
        y += small_font.get_linesize()

    return y


def draw_matrix(screen: pygame.Surface, font: pygame.font.Font) -> None:
    mx = s(PLOT_RIGHT - 170)
    my = s(PLOT_TOP + 8)
    if selected_m == 0:
        screen.blit(font.render("V₀ lines:", True, GRAY), (mx, my))
        entries = v_matrix_entries(0)
        for k, entry in enumerate(entries):
            screen.blit(font.render(f"  {entry}", True, WHITE),
                        (mx, my + font.get_linesize() + s(k * 16)))
        return
    screen.blit(font.render(f"V{selected_m} matrix (diag):", True, GRAY), (mx, my))
    entries = v_matrix_entries(selected_m)
    for k, entry in enumerate(entries):
        color = HIGHLIGHT if k == selected_m else WHITE
        screen.blit(font.render(f" [{k}]={entry}", True, color), (mx, my + font.get_linesize() + s(k * 16)))


def draw_v0_lines(screen: pygame.Surface, color: tuple[int, int, int], width: int) -> None:
    for xr in v0_stationary_xs():
        p_top = world_to_screen(xr, Y_MAX)
        p_bot = world_to_screen(xr, Y_MIN)
        pygame.draw.line(screen, color, p_top, p_bot, width)


def draw_trail(screen: pygame.Surface, canvas_size: tuple[int, int]) -> None:
    if len(trail_history) < 2:
        return
    trail_surf = pygame.Surface(canvas_size, pygame.SRCALPHA)
    for i, frame_segments in enumerate(trail_history):
        alpha = int(TRAIL_ALPHA * i / len(trail_history))
        color = (255, 90, 90, alpha)
        for seg in frame_segments:
            if len(seg) >= 2:
                pygame.draw.lines(trail_surf, color, False, seg, s(1))
    screen.blit(trail_surf, (0, 0))


def all_v_range() -> range:
    start = 0 if show_v0 else 1
    return range(start, max_operator_m() + 1)


def draw_all_v_legend(screen: pygame.Surface, font: pygame.font.Font) -> None:
    x = s(PLOT_LEFT + 10)
    y = s(PLOT_TOP + 28)
    for m in all_v_range():
        color = v_operator_color(m)
        active = m == selected_m
        label = f"V{m}" + (" ◀" if active else "")
        pygame.draw.line(screen, color, (x, y + s(6)), (x + s(18), y + s(6)), s(2 if active else 1))
        screen.blit(font.render(label, True, color if active else GRAY), (x + s(22), y))
        y += font.get_linesize()


def draw_stationary_points(screen: pygame.Surface, font: pygame.font.Font,
                           show_y_labels: bool) -> None:
    for xr in find_stationary_points():
        yr = poly(xr)
        sx, sy = world_to_screen(xr, yr)
        pygame.draw.circle(screen, YELLOW, (sx, sy), s(5))
        pygame.draw.circle(screen, (200, 160, 0), (sx, sy), s(5), s(2))
        if show_y_labels:
            label = font.render(f"y={yr:.2f}", True, YELLOW)
            screen.blit(label, (sx + s(8), sy - label.get_height() // 2))


def add_term(sliders: list[Slider]) -> None:
    global coeffs, selected_m
    new_deg = len(coeffs)
    coeffs.append(0.0)
    ensure_term_meta(new_deg)
    sliders.append(Slider(new_deg, 0.0))
    selected_m = clamp_m(selected_m)


def remove_term(sliders: list[Slider]) -> None:
    global coeffs, selected_m
    if len(coeffs) <= 1:
        return
    removed = len(coeffs) - 1
    coeffs.pop()
    remove_term_meta(removed)
    sliders.pop()
    v_op_colors.pop(removed, None)
    selected_m = clamp_m(selected_m)


def make_ui_rect(x: int, y: int, w: int, h: int) -> pygame.Rect:
    return pygame.Rect(s(x), s(y), s(w), s(h))


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    global playing, selected_m, show_v0, coeffs, trail_history

    pygame.init()
    display = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("V-Operator Explorer")
    canvas = pygame.Surface((s(WIN_W), s(WIN_H)))
    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont("menlo,consolas,monospace", s(15))
    font = pygame.font.SysFont("menlo,consolas,monospace", s(13))
    small_font = pygame.font.SysFont("menlo,consolas,monospace", s(11))

    init_colors()
    selected_m = clamp_m(selected_m)
    sliders = [Slider(k, coeffs[k]) for k in range(len(coeffs))]

    btn_y = 108
    btn_play = Button(make_ui_rect(MARGIN, btn_y, 64, 26), "Play")
    btn_pause = Button(make_ui_rect(MARGIN + 72, btn_y, 64, 26), "Pause")
    btn_plus = Button(make_ui_rect(MARGIN + 156, btn_y, BTN_SIZE, BTN_SIZE), "+")
    btn_minus = Button(make_ui_rect(MARGIN + 192, btn_y, BTN_SIZE, BTN_SIZE), "−")

    chk_y = WIN_H - BOTTOM_RESERVE + 10
    chk_monomials = Checkbox(make_ui_rect(MARGIN, chk_y, PANEL_W - 32, CHECKBOX_H),
                             "Show Monomial Components")
    chk_v_components = Checkbox(make_ui_rect(MARGIN, chk_y + 26, PANEL_W - 32, CHECKBOX_H),
                                "Show V_m Components")
    chk_all_v = Checkbox(make_ui_rect(MARGIN, chk_y + 52, PANEL_W - 32, CHECKBOX_H),
                         "Draw All V_m")
    chk_stationary = Checkbox(make_ui_rect(MARGIN, chk_y + 78, PANEL_W - 32, CHECKBOX_H),
                              "Show Stationary Points")
    chk_matrix = Checkbox(make_ui_rect(MARGIN, chk_y + 104, PANEL_W - 32, CHECKBOX_H),
                         "Show V_m Matrix")
    chk_show_v0 = Checkbox(make_ui_rect(PLOT_LEFT + 10, WIN_H - 62, 160, CHECKBOX_H),
                           "Show V₀")
    chk_trail = Checkbox(make_ui_rect(PLOT_LEFT + 10, WIN_H - 36, 160, CHECKBOX_H),
                         "Show Trail", checked=True)

    t0 = pygame.time.get_ticks()

    running = True
    while running:
        clock.tick(FPS)
        now = (pygame.time.get_ticks() - t0) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    selected_m = clamp_m(selected_m + 1)
                elif event.key == pygame.K_LEFT:
                    selected_m = clamp_m(selected_m - 1)
                elif event.key == pygame.K_UP:
                    add_term(sliders)
                elif event.key == pygame.K_DOWN:
                    remove_term(sliders)
            elif event.type == pygame.MOUSEWHEEL:
                delta = 1 if event.y > 0 else -1
                selected_m = clamp_m(selected_m + delta)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 4:
                selected_m = clamp_m(selected_m + 1)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 5:
                selected_m = clamp_m(selected_m - 1)
            else:
                for sl in sliders:
                    if sl.handle_event(event, RENDER_SCALE):
                        coeffs[sl.degree] = sl.value
                if btn_plus.handle_event(event, RENDER_SCALE):
                    add_term(sliders)
                if btn_minus.handle_event(event, RENDER_SCALE):
                    remove_term(sliders)
                if btn_play.handle_event(event, RENDER_SCALE):
                    playing = True
                if btn_pause.handle_event(event, RENDER_SCALE):
                    playing = False
                chk_monomials.handle_event(event, RENDER_SCALE)
                chk_v_components.handle_event(event, RENDER_SCALE)
                chk_all_v.handle_event(event, RENDER_SCALE)
                chk_stationary.handle_event(event, RENDER_SCALE)
                chk_matrix.handle_event(event, RENDER_SCALE)
                chk_trail.handle_event(event, RENDER_SCALE)
                if chk_show_v0.handle_event(event, RENDER_SCALE):
                    show_v0 = chk_show_v0.checked
                    selected_m = clamp_m(selected_m)

        if playing:
            for k in range(len(coeffs)):
                coeffs[k] = 10 * math.sin(now * speed_omegas[k] + phase_offsets[k])
                sliders[k].value = coeffs[k]

        for sl in sliders:
            sl._update_rect()

        canvas.fill(BG)

        # Left panel
        pygame.draw.rect(canvas, PANEL_BG, (0, 0, s(PANEL_W), s(WIN_H)))
        pygame.draw.line(canvas, GRID, (s(PANEL_W), 0), (s(PANEL_W), s(WIN_H)), s(2))

        draw_formula_panel(canvas, title_font, small_font)

        canvas.blit(small_font.render("Coefficients  [-10, 10]", True, GRAY),
                    (s(SLIDER_LEFT), s(slider_top_y() - 18)))

        for sl in sliders:
            highlighted = selected_m > 0 and sl.degree == selected_m
            sl.draw(canvas, font, highlighted)

        btn_play.draw(canvas, small_font, active=playing)
        btn_pause.draw(canvas, small_font, active=not playing)
        btn_plus.draw(canvas, font)
        btn_minus.draw(canvas, font)

        chk_monomials.draw(canvas, small_font)
        chk_v_components.draw(canvas, small_font)
        chk_all_v.draw(canvas, small_font)
        chk_stationary.draw(canvas, small_font)
        chk_matrix.draw(canvas, small_font)
        chk_show_v0.draw(canvas, small_font)
        chk_trail.draw(canvas, small_font)

        # Plot area
        plot_rect = pygame.Rect(s(PLOT_LEFT), s(PLOT_TOP),
                                s(PLOT_RIGHT - PLOT_LEFT), s(PLOT_BOTTOM - PLOT_TOP))
        pygame.draw.rect(canvas, (22, 22, 34), plot_rect)
        draw_grid(canvas)

        v_segments = [] if selected_m == 0 else sample_curve_segments(v_poly)

        if chk_trail.checked and not chk_all_v.checked and selected_m != 0:
            trail_history.append(v_segments)
            if len(trail_history) > MAX_TRAIL:
                trail_history.pop(0)
            draw_trail(canvas, (s(WIN_W), s(WIN_H)))
        else:
            trail_history.clear()

        if chk_monomials.checked:
            for k, a in enumerate(coeffs):
                if abs(a) < 1e-9:
                    continue
                draw_curve_segments(
                    canvas,
                    sample_curve_segments(lambda x, k=k, a=a: monomial(x, k, a)),
                    term_colors[k], s(1))

        if chk_v_components.checked and selected_m > 0:
            for k, a in enumerate(coeffs):
                coeff = v_coeff(k, a, selected_m)
                if abs(coeff) < 1e-9:
                    continue
                draw_curve_segments(
                    canvas,
                    sample_curve_segments(lambda x, k=k, a=a, m=selected_m: v_monomial(x, k, a, m)),
                    term_colors[k], s(1))

        draw_curve_segments(canvas, sample_curve_segments(poly), BLUE, s(2))

        if chk_all_v.checked:
            for m in all_v_range():
                color = v_operator_color(m)
                width = s(3) if m == selected_m else s(1)
                if m == 0:
                    draw_v0_lines(canvas, color, width)
                else:
                    draw_curve_segments(
                        canvas,
                        sample_curve_segments(lambda x, m=m: v_poly(x, m=m)),
                        color, width)
            draw_all_v_legend(canvas, small_font)
        elif selected_m == 0:
            draw_v0_lines(canvas, v_operator_color(0), s(2))
        else:
            draw_curve_segments(canvas, v_segments, RED, s(2))

        if chk_stationary.checked:
            show_y_labels = not chk_all_v.checked
            draw_stationary_points(canvas, small_font, show_y_labels)

        if chk_matrix.checked:
            draw_matrix(canvas, small_font)

        if not chk_all_v.checked:
            canvas.blit(small_font.render("P(x)", True, BLUE), (s(PLOT_LEFT + 10), s(PLOT_TOP + 8)))
            vm_label = "V₀: x = stationary" if selected_m == 0 else f"V{selected_m}(P)"
            vm_color = v_operator_color(0) if selected_m == 0 else RED
            canvas.blit(small_font.render(vm_label, True, vm_color),
                        (s(PLOT_LEFT + 50), s(PLOT_TOP + 8)))

        scaled = pygame.transform.smoothscale(canvas, (WIN_W, WIN_H))
        display.blit(scaled, (0, 0))
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
