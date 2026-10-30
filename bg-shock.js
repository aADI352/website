// Schlieren shock tube background - planar shock hitting a cylinder
// 2D Euler (MUSCL + HLL + SSP-RK2) in WebGL2
// Adi Atrash, adidi.at
(() => {
if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

const canvas = document.createElement('canvas');
canvas.id = 'shockbg';
canvas.setAttribute('aria-hidden', 'true');
Object.assign(canvas.style, {
    position: 'fixed', inset: '0', width: '100vw', height: '100vh',
    zIndex: '-1', pointerEvents: 'none', opacity: '0', transition: 'opacity 1.2s ease'
});
document.body.prepend(canvas);

const gl = canvas.getContext('webgl2', { antialias: false, depth: false, stencil: false, premultipliedAlpha: true, alpha: true });
if (!gl || !gl.getExtension('EXT_color_buffer_float')) { canvas.remove(); return; }

const GAMMA = 1.4, DT = 0.07;
// incident shock: Rankine-Hugoniot post-shock state into quiescent gas (rho = p = 1)
const MS = 1.7, C0 = Math.sqrt(GAMMA);
const P2 = 1 + 2*GAMMA/(GAMMA + 1)*(MS*MS - 1);
const R2 = (GAMMA + 1)*MS*MS/((GAMMA - 1)*MS*MS + 2);
const U2 = C0*2/(GAMMA + 1)*(MS - 1/MS);
const VS_SHOCK = MS*C0;

const VS = `#version 300 es
void main(){ vec2 p = vec2((gl_VertexID<<1)&2, gl_VertexID&2); gl_Position = vec4(p*2.0-1.0,0.0,1.0); }`;
const HEAD = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform ivec2 uN;
const float G = ${GAMMA.toFixed(2)};
vec4 consv(vec4 w){ return vec4(w.x, w.x*w.y, w.x*w.z, w.w/(G-1.0) + 0.5*w.x*(w.y*w.y+w.z*w.z)); }
`;

const STEP = HEAD + `
uniform sampler2D uU, uB, uM;
uniform float uDt, uA;
uniform int uFinal, uDrive;
uniform vec4 uIn;
out vec4 o;

vec4 prim(vec4 U){
    float r = max(U.x, 1e-3), u = U.y/r, v = U.z/r;
    return vec4(r, u, v, max((G-1.0)*(U.w - 0.5*r*(u*u+v*v)), 1e-3));
}
bool solid(ivec2 p){ return texelFetch(uM, p, 0).r > 0.5; }
vec4 cellAt(ivec2 c, ivec2 off, vec4 wc, out bool mir){
    ivec2 p = c + off;
    mir = false;
    if (p.y < 0 || p.y >= uN.y) mir = true;                       // channel walls top and bottom
    else if (p.x < 0 && uDrive == 1) return uIn;                  // driven inflow behind the shock
    else { p.x = clamp(p.x, 0, uN.x - 1); mir = solid(p); }       // zero-gradient outflow otherwise
    if (!mir) return prim(texelFetch(uU, p, 0));
    vec4 m = wc; if (off.x != 0) m.y = -m.y; if (off.y != 0) m.z = -m.z; return m;
}
float mm(float a, float b){ return (a*b <= 0.0) ? 0.0 : (abs(a) < abs(b) ? a : b); }
vec4 mm4(vec4 a, vec4 b){ return vec4(mm(a.x,b.x), mm(a.y,b.y), mm(a.z,b.z), mm(a.w,b.w)); }
vec4 recL(vec4 a, vec4 b, vec4 c){ return b + 0.5*mm4(b-a, c-b); }
vec4 recR(vec4 b, vec4 c, vec4 d){ return c - 0.5*mm4(c-b, d-c); }
vec4 hll(vec4 L, vec4 R){
    float cL = sqrt(G*L.w/L.x), cR = sqrt(G*R.w/R.x);
    float sL = min(L.y-cL, R.y-cR), sR = max(L.y+cL, R.y+cR);
    vec4 UL = consv(L), UR = consv(R);
    vec4 FL = vec4(UL.y, UL.y*L.y + L.w, UL.z*L.y, (UL.w + L.w)*L.y);
    vec4 FR = vec4(UR.y, UR.y*R.y + R.w, UR.z*R.y, (UR.w + R.w)*R.y);
    if (sL >= 0.0) return FL;
    if (sR <= 0.0) return FR;
    return (sR*FL - sL*FR + sL*sR*(UR-UL))/(sR-sL);
}
void main(){
    ivec2 c = ivec2(gl_FragCoord.xy);
    if (solid(c)) { o = consv(vec4(1.0,0.0,0.0,1.0)); return; }
    vec4 U = texelFetch(uU, c, 0), w = prim(U);
    bool m1, m2;
    vec4 xm1 = cellAt(c, ivec2(-1,0), w, m1), xm2 = cellAt(c, ivec2(-2,0), w, m2); if (m1) xm2 = xm1;
    vec4 xp1 = cellAt(c, ivec2( 1,0), w, m1), xp2 = cellAt(c, ivec2( 2,0), w, m2); if (m1) xp2 = xp1;
    vec4 ym1 = cellAt(c, ivec2(0,-1), w, m1), ym2 = cellAt(c, ivec2(0,-2), w, m2); if (m1) ym2 = ym1;
    vec4 yp1 = cellAt(c, ivec2(0, 1), w, m1), yp2 = cellAt(c, ivec2(0, 2), w, m2); if (m1) yp2 = yp1;
    vec4 F = hll(recL(xm1, w, xp1), recR(w, xp1, xp2)) - hll(recL(xm2, xm1, w), recR(xm1, w, xp1));
    vec4 Gp = hll(recL(ym1.xzyw, w.xzyw, yp1.xzyw), recR(w.xzyw, yp1.xzyw, yp2.xzyw));
    vec4 Gm = hll(recL(ym2.xzyw, ym1.xzyw, w.xzyw), recR(ym1.xzyw, w.xzyw, yp1.xzyw));
    vec4 Un = U - uDt*(F + (Gp - Gm).xzyw);
    Un = uA*texelFetch(uB, c, 0) + (1.0-uA)*Un;
    if (uFinal == 1) {
        Un = mix(Un, consv(vec4(1.0,0.0,0.0,1.0)), 0.0006);   // slow relaxation to ambient
        vec4 q = prim(Un);
        if (any(isnan(Un)) || Un.x < 0.02) q = vec4(1.0,0.0,0.0,1.0);
        Un = consv(q);
    }
    o = Un;
}`;

const DISP = HEAD + `
uniform sampler2D uU, uM;
uniform vec3 uInk; uniform float uK, uAlpha;
out vec4 col;
float rho(ivec2 p){ return texelFetch(uU, clamp(p, ivec2(0), uN-1), 0).x; }
void main(){
    ivec2 c = ivec2(gl_FragCoord.xy);
    float mk = texelFetch(uM, c, 0).r;
    if (mk > 0.5) { col = vec4(0.0); return; }
    float m8 = mk*255.0;
    float dim = m8 > 100.5 ? 1.0 - clamp((m8 - 100.0)/27.0, 0.0, 1.0)   // hide ring around the photo
                           : 1.0 - 0.75*clamp(m8/100.0, 0.0, 1.0);        // fade behind text
    float gx = rho(c+ivec2(1,0)) - rho(c-ivec2(1,0));
    float gy = rho(c+ivec2(0,1)) - rho(c-ivec2(0,1));
    float g = length(vec2(gx, gy))/rho(c);
    float a = dim*uAlpha*(1.0 - exp(-uK*max(g - 0.012, 0.0)));
    col = vec4(uInk*a, a);
}`;

function prog(fs) {
    const p = gl.createProgram();
    for (const [t, src] of [[gl.VERTEX_SHADER, VS], [gl.FRAGMENT_SHADER, fs]]) {
        const s = gl.createShader(t); gl.shaderSource(s, src); gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
        gl.attachShader(p, s);
    }
    gl.linkProgram(p);
    const u = {}, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) { const nm = gl.getActiveUniform(p, i).name; u[nm] = gl.getUniformLocation(p, nm); }
    return { p, u };
}
let P;
try { P = { step: prog(STEP), disp: prog(DISP) }; }
catch (e) { console.warn('shock bg disabled:', e); canvas.remove(); return; }
gl.bindVertexArray(gl.createVertexArray());

let NX = 0, NY = 0, CELL = 4, A, S, B, maskTex, mask;
const ambient = (n) => { const d = new Float32Array(n*4); for (let k = 0; k < n; k++) { d[4*k] = 1; d[4*k+3] = 1/(GAMMA-1); } return d; };

function tex(fmt, ifmt, type, data, w, h) {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, ifmt, w, h, 0, fmt, type, data);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]])
        gl.texParameteri(gl.TEXTURE_2D, k, v);
    return t;
}
function target(data) {
    const t = tex(gl.RGBA, gl.RGBA32F, gl.FLOAT, data, NX, NY), fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
    return { t, fb };
}
function freeAll() {
    for (const T of [A, S, B]) if (T) { gl.deleteTexture(T.t); gl.deleteFramebuffer(T.fb); }
    if (maskTex) gl.deleteTexture(maskTex);
}
function setup() {
    const w = window.innerWidth, h = window.innerHeight;
    // aim for ~220k cells max, never finer than 2 css px
    CELL = Math.max(2, Math.ceil(Math.sqrt(w*h/220000)));
    NX = Math.ceil(w/CELL); NY = Math.ceil(h/CELL);
    canvas.width = NX; canvas.height = NY;
    freeAll();
    const amb = ambient(NX*NY);
    A = target(amb); S = target(null); B = target(amb);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    mask = new Uint8Array(NX*NY);
    maskTex = tex(gl.RED, gl.R8, gl.UNSIGNED_BYTE, mask, NX, NY);
    maskDirty = true;
}

// content blocks act as solid walls
const SOLID_SEL = '.pfp';
const DIM_SEL = '.hero p, .blghd, .blogentry';
let maskDirty = true;
function buildMask() {
    mask.fill(0);
    // soft fade zones behind text (mask 1..100 = fluid, dimmed; 255 = solid wall)
    const F = 36/CELL, H = window.innerHeight;
    for (const el of document.querySelectorAll(DIM_SEL)) {
        const r = el.getBoundingClientRect();
        if (r.bottom < -48 || r.top > H + 48) continue;
        const x0 = r.left/CELL, x1 = r.right/CELL, y0 = (H - r.bottom)/CELL, y1 = (H - r.top)/CELL;
        const i0 = Math.max(0, Math.floor(x0 - F)), i1 = Math.min(NX-1, Math.ceil(x1 + F));
        const j0 = Math.max(0, Math.floor(y0 - F)), j1 = Math.min(NY-1, Math.ceil(y1 + F));
        for (let j = j0; j <= j1; j++) {
            const dy = Math.max(y0 - j - 0.5, j + 0.5 - y1, 0);
            for (let i = i0; i <= i1; i++) {
                const dx = Math.max(x0 - i - 0.5, i + 0.5 - x1, 0);
                const w = 1 - Math.min(Math.hypot(dx, dy)/F, 1);
                const v = Math.round(100*w*w*(3 - 2*w));
                const k = j*NX + i;
                if (v > mask[k]) mask[k] = v;
            }
        }
    }
    for (const el of document.querySelectorAll(SOLID_SEL)) {
        const r0 = el.getBoundingClientRect(), pad = el.classList.contains('pfp') ? 0 : 8;
        const r = { left: r0.left, right: r0.right, top: r0.top - pad, bottom: r0.bottom + pad, width: r0.width };
        if (r.bottom < 0 || r.top > window.innerHeight) continue;
        const i0 = Math.max(0, Math.floor(r.left/CELL)), i1 = Math.min(NX-1, Math.floor(r.right/CELL));
        const j0 = Math.max(0, Math.floor((window.innerHeight - r.bottom)/CELL)), j1 = Math.min(NY-1, Math.floor((window.innerHeight - r.top)/CELL));
        const round = el.classList.contains('pfp');
        const cx = (r.left + r.right)/2/CELL, cy = (window.innerHeight - (r.top + r.bottom)/2)/CELL, rad = r.width/2/CELL;
        const ring = 14/CELL;
        for (let j = Math.max(0, j0 - Math.ceil(ring)); j <= Math.min(NY-1, j1 + Math.ceil(ring)); j++)
            for (let i = Math.max(0, i0 - Math.ceil(ring)); i <= Math.min(NX-1, i1 + Math.ceil(ring)); i++) {
                const d = Math.hypot(i + 0.5 - cx, j + 0.5 - cy) - rad;
                const k = j*NX + i;
                if (d < -0.5) { mask[k] = 255; continue; }
                // hide the stair-stepped wall in a thin feathered ring
                const h = 1 - Math.min(Math.max(d + 0.5, 0)/ring, 1);
                const v = 100 + Math.round(27*h*h*(3 - 2*h));
                if (v > 100 && v > mask[k]) mask[k] = v;
            }
    }
    gl.bindTexture(gl.TEXTURE_2D, maskTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, NX, NY, gl.RED, gl.UNSIGNED_BYTE, mask);
    maskDirty = false;
}

function bind(pr, name, unit, t) { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, t); gl.uniform1i(pr.u[name], unit); }

function step(n, drive) {
    const pr = P.step; gl.useProgram(pr.p); gl.viewport(0, 0, NX, NY);
    gl.uniform2i(pr.u.uN, NX, NY); gl.uniform1f(pr.u.uDt, DT);
    gl.uniform1i(pr.u.uDrive, drive ? 1 : 0); gl.uniform4f(pr.u.uIn, R2, U2, 0, P2);
    bind(pr, 'uM', 2, maskTex);
    for (let k = 0; k < n; k++) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, S.fb);
        bind(pr, 'uU', 0, A.t); bind(pr, 'uB', 1, A.t);
        gl.uniform1f(pr.u.uA, 0); gl.uniform1i(pr.u.uFinal, 0);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.bindFramebuffer(gl.FRAMEBUFFER, B.fb);
        bind(pr, 'uU', 0, S.t); bind(pr, 'uB', 1, A.t);
        gl.uniform1f(pr.u.uA, 0.5); gl.uniform1i(pr.u.uFinal, 1);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        [A, B] = [B, A];
    }
}
function draw() {
    const pr = P.disp; gl.useProgram(pr.p);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, NX, NY);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    bind(pr, 'uU', 0, A.t); bind(pr, 'uM', 1, maskTex);
    gl.uniform2i(pr.u.uN, NX, NY);
    gl.uniform3f(pr.u.uInk, 0.12, 0.14, 0.18);
    gl.uniform1f(pr.u.uK, 12.0); gl.uniform1f(pr.u.uAlpha, 0.65);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
}

window.addEventListener('scroll', () => { maskDirty = true; }, { passive: true });
let rt;
window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { setup(); tc = 0; }, 150); });

setup();
buildMask();
// shock tube cycle, in sim time: drive the left boundary, then let it run out and reset
let tc = 0;
function cycle() {
    const cross = NX/VS_SHOCK;
    return { drive: 0.45*cross, period: 1.9*cross };
}
function frame() {
    if (!document.hidden) {
        if (maskDirty) buildMask();
        // ~9 s for the shock to cross the screen
        const n = Math.min(12, Math.max(2, Math.round(NX/VS_SHOCK/9/60/DT)));
        const cy = cycle();
        for (let k = 0; k < n; k++) {
            step(1, tc < cy.drive);
            tc += DT;
            if (tc > cy.period) tc = 0;
        }
        draw();
    }
    requestAnimationFrame(frame);
}
requestAnimationFrame(() => { canvas.style.opacity = '1'; frame(); });
window.__shock = { step, draw, buildMask, get NX(){ return NX; } };
})();
