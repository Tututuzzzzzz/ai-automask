/* Công cụ duyệt nhãn phân vùng cho bộ dữ liệu cốc sứ.
 *
 * Vì sao cần: nhãn do AI sinh là *đề xuất*, chưa phải ground truth. Kế hoạch
 * thực tập yêu cầu "bộ dữ liệu đã gán nhãn", nghĩa là phải có người xác nhận.
 * Thực tế trên ảnh nền trắng thì phần lớn chỉ cần bấm "Nhận"; số ít cần sửa vài
 * nét cọ. Công cụ này tối ưu cho đúng nhịp làm việc đó: một phím cho ca dễ, cọ
 * vẽ cho ca khó.
 *
 * Mask được chỉnh ngay trên canvas ở đúng độ phân giải gốc rồi POST về server,
 * nên nhãn đã duyệt luôn khớp pixel với ảnh - đúng ràng buộc của toàn dự án.
 */
'use strict';

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

const S = {
  dataset: null,
  items: [],
  filter: 'all',
  idx: -1,
  tool: 'pan',
  mode: 'overlay',
  brush: 34,
  opacity: 0.55,
  // Ảnh gốc + mask ở độ phân giải gốc; mọi chỉnh sửa diễn ra trên maskCanvas.
  img: null,
  maskCanvas: null,
  maskCtx: null,
  history: [],
  dirty: false,
  view: { scale: 1, ox: 0, oy: 0 },
  drawing: false,
  spaceDown: false,
};

const canvas = $('#canvas');
const ctx = canvas.getContext('2d');

/* ------------------------------------------------------------------ tải dữ liệu */
async function boot() {
  try {
    const d = await (await fetch('/v1/labeling/dataset')).json();
    S.dataset = d.dataset;
    S.items = d.items;
    $('#dsname').textContent = `${d.dataset} · ${d.items.length} ảnh`;
    renderList();
    renderCounters();
    if (S.items.length) select(0);
  } catch (e) {
    $('#hint').textContent = 'Không tải được dataset: ' + e.message;
  }
}

function statusOf(it) { return it.review_status || ''; }

function visible() {
  if (S.filter === 'all') return S.items;
  if (S.filter === 'todo') return S.items.filter((i) => !statusOf(i));
  return S.items.filter((i) => statusOf(i) === S.filter);
}

function renderCounters() {
  const c = { accept: 0, fix: 0, reject: 0, todo: 0 };
  S.items.forEach((i) => { const s = statusOf(i); if (s) c[s]++; else c.todo++; });
  $('#counters').innerHTML =
    `<span class="c"><b>${c.todo}</b> chưa duyệt</span>` +
    `<span class="c accept"><b>${c.accept}</b> nhận</span>` +
    `<span class="c fix"><b>${c.fix}</b> đã sửa</span>` +
    `<span class="c reject"><b>${c.reject}</b> loại</span>`;
}

function renderList() {
  const list = visible();
  $('#items').innerHTML = list.map((it) => {
    const st = statusOf(it);
    const i = S.items.indexOf(it);
    return `<li data-i="${i}" class="${i === S.idx ? 'sel' : ''}">
      <span class="dot ${st}"></span>
      <img loading="lazy" src="/v1/labeling/file/${encodeURIComponent(it.id)}/overlay" alt="">
      <span class="n"><span class="t">${it.id}</span>
      <span class="s">${it.verdict || '?'} · ${Number(it.confidence || 0).toFixed(3)}</span></span>
    </li>`;
  }).join('');
  $$('#items li').forEach((li) =>
    li.addEventListener('click', () => select(Number(li.dataset.i))));
}

/* --------------------------------------------------------------- chọn 1 ảnh */
async function select(i) {
  if (i < 0 || i >= S.items.length) return;
  if (S.dirty && !confirm('Bản sửa chưa lưu sẽ mất. Chuyển ảnh?')) return;
  S.idx = i;
  const it = S.items[i];
  $('#hint').hidden = true;
  renderList();

  const [img, mask] = await Promise.all([
    loadImage(`/v1/labeling/file/${encodeURIComponent(it.id)}/image`),
    loadImage(`/v1/labeling/file/${encodeURIComponent(it.id)}/label`),
  ]);
  S.img = img;
  S.maskCanvas = document.createElement('canvas');
  S.maskCanvas.width = img.naturalWidth;
  S.maskCanvas.height = img.naturalHeight;
  S.maskCtx = S.maskCanvas.getContext('2d', { willReadFrequently: true });
  S.maskCtx.drawImage(mask, 0, 0, S.maskCanvas.width, S.maskCanvas.height);
  S.history = [];
  S.dirty = false;

  $('#meta').innerHTML =
    `<b>${it.id}</b> · ${img.naturalWidth}×${img.naturalHeight} · ` +
    `base ${it.base_code} · màu ${it.color} · mặt ${it.side} · ` +
    `máy chấm <b>${it.verdict}</b> ${Number(it.confidence || 0).toFixed(3)}`;

  fitView();
  draw();
}

function loadImage(src) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error('không tải được ' + src));
    im.src = src;
  });
}

/* ------------------------------------------------------------------ vẽ canvas */
function fitView() {
  const wrap = $('#canvaswrap');
  const cw = wrap.clientWidth - 32, ch = wrap.clientHeight - 32;
  const s = Math.min(cw / S.img.naturalWidth, ch / S.img.naturalHeight, 1);
  S.view = { scale: s, ox: 0, oy: 0 };
  canvas.width = Math.round(S.img.naturalWidth * s);
  canvas.height = Math.round(S.img.naturalHeight * s);
}

function draw() {
  if (!S.img) return;
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  if (S.mode !== 'mask') ctx.drawImage(S.img, 0, 0, w, h);

  if (S.mode === 'mask') {
    ctx.drawImage(S.maskCanvas, 0, 0, w, h);
  } else if (S.mode === 'overlay') {
    // Tô xanh vùng mask: mắt người bắt lỗi viền trên nền màu tốt hơn nhiều so
    // với so sánh hai ảnh xám cạnh nhau.
    //
    // Alpha phải lấy từ ĐỘ SÁNG của mask, không phải từ kênh alpha của nó. Mask
    // là ảnh xám đục hoàn toàn (nền đen vẫn có alpha 255), nên phép 'source-in'
    // sẽ giữ nguyên cả khung và tô xanh toàn bộ ảnh.
    const tint = document.createElement('canvas');
    tint.width = w; tint.height = h;
    const tctx = tint.getContext('2d', { willReadFrequently: true });
    tctx.drawImage(S.maskCanvas, 0, 0, w, h);
    const px = tctx.getImageData(0, 0, w, h);
    const d = px.data;
    for (let i = 0; i < d.length; i += 4) {
      const lum = d[i + 3] === 0 ? 0 : d[i];   // vùng chưa vẽ coi như nền
      d[i] = 46; d[i + 1] = 204; d[i + 2] = 113;
      d[i + 3] = lum;
    }
    tctx.putImageData(px, 0, 0);
    ctx.globalAlpha = S.opacity;
    ctx.drawImage(tint, 0, 0);
    ctx.globalAlpha = 1;
  }
}

/* --------------------------------------------------------------------- cọ vẽ */
function canvasToMask(ev) {
  const r = canvas.getBoundingClientRect();
  return {
    x: (ev.clientX - r.left) / r.width * S.maskCanvas.width,
    y: (ev.clientY - r.top) / r.height * S.maskCanvas.height,
  };
}

function pushHistory() {
  // Giữ tối đa 20 bước; ảnh 1200×1200 nên bộ nhớ vẫn nhẹ.
  S.history.push(S.maskCtx.getImageData(0, 0, S.maskCanvas.width, S.maskCanvas.height));
  if (S.history.length > 20) S.history.shift();
}

function paint(a, b) {
  const c = S.maskCtx;
  c.save();
  c.globalCompositeOperation = 'source-over';
  // Mask là ảnh xám: vẽ trắng để thêm, đen để xoá.
  c.strokeStyle = S.tool === 'add' ? '#ffffff' : '#000000';
  c.fillStyle = c.strokeStyle;
  c.lineWidth = S.brush;
  c.lineCap = 'round';
  c.lineJoin = 'round';
  c.beginPath();
  c.moveTo(a.x, a.y);
  c.lineTo(b.x, b.y);
  c.stroke();
  c.restore();
}

let last = null;
canvas.addEventListener('pointerdown', (ev) => {
  if (!S.img || S.tool === 'pan' || S.spaceDown) return;
  canvas.setPointerCapture(ev.pointerId);
  pushHistory();
  S.drawing = true;
  last = canvasToMask(ev);
  paint(last, last);
  S.dirty = true;
  draw();
});
canvas.addEventListener('pointermove', (ev) => {
  if (!S.drawing) return;
  const p = canvasToMask(ev);
  paint(last, p);
  last = p;
  draw();
});
['pointerup', 'pointercancel', 'pointerleave'].forEach((e) =>
  canvas.addEventListener(e, () => { S.drawing = false; last = null; }));

function undo() {
  const prev = S.history.pop();
  if (!prev) return;
  S.maskCtx.putImageData(prev, 0, 0);
  draw();
}

async function resetToAuto() {
  if (!S.items[S.idx]) return;
  const mask = await loadImage(
    `/v1/labeling/file/${encodeURIComponent(S.items[S.idx].id)}/label?auto=1&t=${Date.now()}`);
  pushHistory();
  S.maskCtx.clearRect(0, 0, S.maskCanvas.width, S.maskCanvas.height);
  S.maskCtx.drawImage(mask, 0, 0, S.maskCanvas.width, S.maskCanvas.height);
  S.dirty = true;
  draw();
}

/* ------------------------------------------------------------------ lưu quyết định */
async function decide(status) {
  const it = S.items[S.idx];
  if (!it) return;
  const body = { id: it.id, status };
  if (status === 'fix') {
    // Nhị phân hoá trước khi gửi: nhãn huấn luyện phải là mask sạch, không phải
    // vệt cọ nửa trong suốt.
    body.mask_png = binarisedMaskDataURL();
  }
  const r = await fetch('/v1/labeling/decide', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) { alert('Lưu thất bại: ' + (await r.text())); return; }
  it.review_status = status;
  S.dirty = false;
  flashSaved();
  renderCounters();
  renderList();
  next();
}

function binarisedMaskDataURL() {
  const w = S.maskCanvas.width, h = S.maskCanvas.height;
  const out = document.createElement('canvas');
  out.width = w; out.height = h;
  const octx = out.getContext('2d');
  const src = S.maskCtx.getImageData(0, 0, w, h);
  const dst = octx.createImageData(w, h);
  for (let i = 0; i < src.data.length; i += 4) {
    // Kênh alpha của canvas cũng phải tính đến: vùng chưa vẽ có alpha = 0.
    const lum = src.data[i + 3] === 0 ? 0 : src.data[i];
    const v = lum >= 128 ? 255 : 0;
    dst.data[i] = dst.data[i + 1] = dst.data[i + 2] = v;
    dst.data[i + 3] = 255;
  }
  octx.putImageData(dst, 0, 0);
  return out.toDataURL('image/png');
}

let savedTimer = null;
function flashSaved() {
  const el = $('#saved');
  el.textContent = 'đã lưu ✓';
  el.classList.add('on');
  clearTimeout(savedTimer);
  savedTimer = setTimeout(() => { el.classList.remove('on'); el.textContent = 'đã lưu'; }, 1400);
}

function next() {
  const list = visible();
  if (!list.length) { $('#hint').hidden = false; $('#hint').textContent = 'Hết ảnh trong bộ lọc này'; return; }
  const cur = S.items[S.idx];
  let j = list.indexOf(cur);
  // Ở bộ lọc "chưa duyệt", ảnh vừa xử lý rời khỏi danh sách nên vị trí j giữ nguyên.
  if (j === -1) j = Math.min(0, list.length - 1);
  else j = Math.min(j + 1, list.length - 1);
  select(S.items.indexOf(list[j]));
}
function prev() {
  const list = visible();
  const j = Math.max(0, list.indexOf(S.items[S.idx]) - 1);
  if (list[j]) select(S.items.indexOf(list[j]));
}

/* ------------------------------------------------------------------- điều khiển */
$('#filters').addEventListener('click', (e) => {
  const b = e.target.closest('.chip'); if (!b) return;
  S.filter = b.dataset.f;
  $$('#filters .chip').forEach((c) => c.classList.toggle('active', c === b));
  renderList();
});
$('#modes').addEventListener('click', (e) => {
  const b = e.target.closest('.chip'); if (!b) return;
  S.mode = b.dataset.mode;
  $$('#modes .chip').forEach((c) => c.classList.toggle('active', c === b));
  draw();
});
$('#opacity').addEventListener('input', (e) => { S.opacity = e.target.value / 100; draw(); });
$('#brush').addEventListener('input', (e) => {
  S.brush = Number(e.target.value); $('#brushval').textContent = S.brush;
});
$$('.tool[data-tool]').forEach((b) => b.addEventListener('click', () => setTool(b.dataset.tool)));
function setTool(t) {
  S.tool = t;
  $$('.tool[data-tool]').forEach((b) => b.classList.toggle('active', b.dataset.tool === t));
  $('#canvaswrap').classList.toggle('panning', t === 'pan');
}
$('#undo').addEventListener('click', undo);
$('#reset').addEventListener('click', resetToAuto);
$('#btn-accept').addEventListener('click', () => decide('accept'));
$('#btn-fix').addEventListener('click', () => decide('fix'));
$('#btn-reject').addEventListener('click', () => decide('reject'));
window.addEventListener('resize', () => { if (S.img) { fitView(); draw(); } });

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.code === 'Space') { S.spaceDown = true; return; }
  if (e.ctrlKey && e.key.toLowerCase() === 'z') { e.preventDefault(); undo(); return; }
  switch (e.key.toLowerCase()) {
    case 'a': decide('accept'); break;
    case 's': decide('fix'); break;
    case 'r': decide('reject'); break;
    case 'b': setTool('add'); break;
    case 'e': setTool('erase'); break;
    case '[': S.brush = Math.max(4, S.brush - 6); $('#brush').value = S.brush; $('#brushval').textContent = S.brush; break;
    case ']': S.brush = Math.min(160, S.brush + 6); $('#brush').value = S.brush; $('#brushval').textContent = S.brush; break;
    case 'arrowright': next(); break;
    case 'arrowleft': prev(); break;
  }
});
document.addEventListener('keyup', (e) => { if (e.code === 'Space') S.spaceDown = false; });
window.addEventListener('beforeunload', (e) => { if (S.dirty) { e.preventDefault(); e.returnValue = ''; } });

boot();
