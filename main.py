import math
import random
import pygame
import sys

# ── Display (logical window smaller; supersampled for sharpness) ───────────────
WIN_W, WIN_H = 1100, 680
RENDER_SCALE = 2
FPS = 30

# ── Plot ───────────────────────────────────────────────────────────────────────
PANEL_W = 420
X_MIN, X_MAX = -10, 10
Y_MIN, Y_MAX = -11, 11
PLOT_LEFT = PANEL_W
PLOT_RIGHT = WIN_W - 24
PLOT_TOP, PLOT_BOTTOM = 36, WIN_H - 24
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
selected_i = 1
term_colors: dict[int, tuple[int, int, int]] = {}
phase_offsets: dict[int, float] = {}
speed_omegas: dict[int, float] = {}
v_op_colors: dict[int, tuple[int, int, int]] = {}
trail_history: list[list[tuple[float, float]]] = []
MAX_TRAIL = 500


def s(v: float | int) -> int:
    return int(v * RENDER_SCALE)


def max_operator_i() -> int:
    return max(1, len(coeffs) - 1)


def clamp_i(i: int) -> int:
    return max(1, min(i, max_operator_i()))


def random_term_color() -> tuple[int, int, int]:
    return tuple(random.randint(80, 255) for _ in range(3))


def v_operator_color(i: int) -> tuple[int, int, int]:
    if i not in v_op_colors:
        hue = (i * 47) % 360
        c = pygame.Color(0)
        c.hsla = (hue, 72, 58, 100)
        v_op_colors[i] = (max(c.r, 80), max(c.g, 80), max(c.b, 80))
    return v_op_colors[i]


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


def v_coeff(k: int, a: float, i: int) -> float:
    if i == 0:
        return 0.0
    return (1 - k / i) * a


def v_poly(x: float, c: list[float] | None = None, i: int | None = None) -> float:
    c = coeffs if c is None else c
    i = selected_i if i is None else i
    if i == 0:
        return 0.0
    return sum(v_coeff(k, a, i) * x**k for k, a in enumerate(c))


def monomial(x: float, k: int, a: float) -> float:
    return a * x**k


def v_monomial(x: float, k: int, a: float, i: int) -> float:
    return v_coeff(k, a, i) * x**k


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


def v_matrix_entries(i: int) -> list[str]:
    n = len(coeffs)
    if i == 0:
        return ["—"] * n
    return [f"{1 - k / i:.3g}" for k in range(n)]


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
def sample_curve(fn, num_points: int | None = None) -> list[tuple[int, int]]:
    num_points = num_points or PLOT_WIDTH
    points = []
    for px in range(num_points):
        x = X_MIN + px / max(num_points - 1, 1) * (X_MAX - X_MIN)
        y = fn(x)
        if Y_MIN - 5 <= y <= Y_MAX + 5:
            points.append(world_to_screen(x, y))
    return points


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
    highlight_k = selected_i if selected_i <= len(coeffs) - 1 else None
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
    op_text = f"Current Operator: V{selected_i}   (scroll ±1, max V{max_operator_i()})"
    screen.blit(small_font.render(op_text, True, HIGHLIGHT), (s(SLIDER_LEFT), y))
    y += small_font.get_linesize() + s(4)

    if highlight_k is not None and highlight_k < len(coeffs):
        elim = small_font.render(f"a{highlight_k} is eliminated by V{selected_i}", True, HIGHLIGHT)
        screen.blit(elim, (s(SLIDER_LEFT), y))
        y += small_font.get_linesize()

    return y


def draw_matrix(screen: pygame.Surface, font: pygame.font.Font) -> None:
    mx = s(PLOT_RIGHT - 170)
    my = s(PLOT_TOP + 8)
    screen.blit(font.render(f"V{selected_i} matrix (diag):", True, GRAY), (mx, my))
    entries = v_matrix_entries(selected_i)
    for k, entry in enumerate(entries):
        color = HIGHLIGHT if k == selected_i else WHITE
        screen.blit(font.render(f" [{k}]={entry}", True, color), (mx, my + font.get_linesize() + s(k * 16)))


def draw_trail(screen: pygame.Surface, canvas_size: tuple[int, int]) -> None:
    if len(trail_history) < 2:
        return
    trail_surf = pygame.Surface(canvas_size, pygame.SRCALPHA)
    for i, frame_points in enumerate(trail_history):
        alpha = int(TRAIL_ALPHA * i / len(trail_history))
        if len(frame_points) >= 2:
            pygame.draw.lines(trail_surf, (255, 90, 90, alpha), False, frame_points, s(1))
    screen.blit(trail_surf, (0, 0))


def draw_all_v_legend(screen: pygame.Surface, font: pygame.font.Font) -> None:
    x = s(PLOT_LEFT + 10)
    y = s(PLOT_TOP + 28)
    for i in range(1, max_operator_i() + 1):
        color = v_operator_color(i)
        active = i == selected_i
        label = f"V{i}" + (" ◀" if active else "")
        pygame.draw.line(screen, color, (x, y + s(6)), (x + s(18), y + s(6)), s(2 if active else 1))
        screen.blit(font.render(label, True, color if active else GRAY), (x + s(22), y))
        y += font.get_linesize()


def make_ui_rect(x: int, y: int, w: int, h: int) -> pygame.Rect:
    return pygame.Rect(s(x), s(y), s(w), s(h))


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    global playing, selected_i, coeffs, trail_history

    pygame.init()
    display = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("V-Operator Explorer")
    canvas = pygame.Surface((s(WIN_W), s(WIN_H)))
    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont("menlo,consolas,monospace", s(15))
    font = pygame.font.SysFont("menlo,consolas,monospace", s(13))
    small_font = pygame.font.SysFont("menlo,consolas,monospace", s(11))

    init_colors()
    selected_i = clamp_i(selected_i)
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
                                "Show Vᵢ Components")
    chk_all_v = Checkbox(make_ui_rect(MARGIN, chk_y + 52, PANEL_W - 32, CHECKBOX_H),
                         "Draw All Vᵢ")
    chk_stationary = Checkbox(make_ui_rect(MARGIN, chk_y + 78, PANEL_W - 32, CHECKBOX_H),
                              "Show Stationary Points")
    chk_matrix = Checkbox(make_ui_rect(MARGIN, chk_y + 104, PANEL_W - 32, CHECKBOX_H),
                         "Show Vᵢ Matrix")
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
            elif event.type == pygame.MOUSEWHEEL:
                delta = 1 if event.y > 0 else -1
                selected_i = clamp_i(selected_i + delta)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 4:
                selected_i = clamp_i(selected_i + 1)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 5:
                selected_i = clamp_i(selected_i - 1)
            else:
                for sl in sliders:
                    if sl.handle_event(event, RENDER_SCALE):
                        coeffs[sl.degree] = sl.value
                if btn_plus.handle_event(event, RENDER_SCALE):
                    new_deg = len(coeffs)
                    coeffs.append(0.0)
                    ensure_term_meta(new_deg)
                    sliders.append(Slider(new_deg, 0.0))
                    selected_i = clamp_i(selected_i)
                if btn_minus.handle_event(event, RENDER_SCALE) and len(coeffs) > 1:
                    removed = len(coeffs) - 1
                    coeffs.pop()
                    remove_term_meta(removed)
                    sliders.pop()
                    v_op_colors.pop(removed, None)
                    selected_i = clamp_i(selected_i)
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
            highlighted = sl.degree == selected_i
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
        chk_trail.draw(canvas, small_font)

        # Plot area
        plot_rect = pygame.Rect(s(PLOT_LEFT), s(PLOT_TOP),
                                s(PLOT_RIGHT - PLOT_LEFT), s(PLOT_BOTTOM - PLOT_TOP))
        pygame.draw.rect(canvas, (22, 22, 34), plot_rect)
        draw_grid(canvas)

        v_points = sample_curve(v_poly)

        if chk_trail.checked and not chk_all_v.checked:
            trail_history.append(v_points)
            if len(trail_history) > MAX_TRAIL:
                trail_history.pop(0)
            draw_trail(canvas, (s(WIN_W), s(WIN_H)))
        else:
            trail_history.clear()

        if chk_monomials.checked:
            for k, a in enumerate(coeffs):
                if abs(a) < 1e-9:
                    continue
                pts = sample_curve(lambda x, k=k, a=a: monomial(x, k, a))
                if len(pts) >= 2:
                    pygame.draw.lines(canvas, term_colors[k], False, pts, s(1))

        if chk_v_components.checked:
            for k, a in enumerate(coeffs):
                coeff = v_coeff(k, a, selected_i)
                if abs(coeff) < 1e-9:
                    continue
                pts = sample_curve(lambda x, k=k, a=a, i=selected_i: v_monomial(x, k, a, i))
                if len(pts) >= 2:
                    pygame.draw.lines(canvas, term_colors[k], False, pts, s(1))

        p_points = sample_curve(poly)
        if len(p_points) >= 2:
            pygame.draw.lines(canvas, BLUE, False, p_points, s(2))

        if chk_all_v.checked:
            for i in range(1, max_operator_i() + 1):
                color = v_operator_color(i)
                pts = sample_curve(lambda x, i=i: v_poly(x, i=i))
                if len(pts) >= 2:
                    width = s(3) if i == selected_i else s(1)
                    pygame.draw.lines(canvas, color, False, pts, width)
            draw_all_v_legend(canvas, small_font)
        elif len(v_points) >= 2:
            pygame.draw.lines(canvas, RED, False, v_points, s(2))

        if chk_stationary.checked:
            for xr in find_stationary_points():
                yr = poly(xr)
                sx, sy = world_to_screen(xr, yr)
                pygame.draw.circle(canvas, YELLOW, (sx, sy), s(5))
                pygame.draw.circle(canvas, (200, 160, 0), (sx, sy), s(5), s(2))

        if chk_matrix.checked:
            draw_matrix(canvas, small_font)

        if not chk_all_v.checked:
            canvas.blit(small_font.render("P(x)", True, BLUE), (s(PLOT_LEFT + 10), s(PLOT_TOP + 8)))
            canvas.blit(small_font.render(f"V{selected_i}(P)", True, RED),
                        (s(PLOT_LEFT + 50), s(PLOT_TOP + 8)))

        scaled = pygame.transform.smoothscale(canvas, (WIN_W, WIN_H))
        display.blit(scaled, (0, 0))
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
