"""
Seoul Sound Map — demo interattiva
Carica un MIDI: la demo estrae la partitura, sintetizza un audio di lavoro,
e mostra la visualizzazione interattiva (curve sonore dentro la mappa di Seoul).
Puoi anche caricare un audio alternativo (es. export dal tuo DAW) da usare al
posto di quello sintetizzato.

Avvio:
    pip install gradio pretty_midi numpy soundfile
    python3 app.py
"""

import base64
import io
import json
import os
import tempfile

import gradio as gr
import numpy as np
import pretty_midi
import soundfile as sf

SR = 22050
import json as _json
_masks_path = os.path.join(os.path.dirname(__file__), "all_masks.json")
if not os.path.exists(_masks_path):
    _masks_path = "all_masks.json"
ALL_MASKS = _json.load(open(_masks_path))
_labels_path = os.path.join(os.path.dirname(__file__), "mask_labels.json")
if not os.path.exists(_labels_path):
    _labels_path = "mask_labels.json"
MASK_LABELS = _json.load(open(_labels_path))
_curated = [k for k in ["seoul","roma","korea","italy"] if k in MASK_LABELS]
_kr = sorted([k for k in MASK_LABELS if k.startswith("krcity_")], key=lambda k: MASK_LABELS[k])
_co = sorted([k for k in MASK_LABELS if k.startswith("country_")], key=lambda k: MASK_LABELS[k])
_other = [k for k in MASK_LABELS if k not in _curated and k not in _kr and k not in _co]
MENU_CHOICES = [MASK_LABELS[k] for k in (_curated + _kr + _co + _other)]


# ---------- estrazione note dal MIDI ----------
def extract_score(midi_path, max_seconds=None):
    pm = pretty_midi.PrettyMIDI(midi_path)
    full_duration = float(pm.get_end_time())
    limit = full_duration if max_seconds is None else min(full_duration, max_seconds)
    events = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.start >= limit:
                continue  # oltre il limite: scarto
            events.append({
                "time": round(float(n.start), 4),
                "freq": round(float(pretty_midi.note_number_to_hz(n.pitch)), 2),
                "amplitude": round(n.velocity / 127.0, 4),
                # taglio anche la durata se la nota sfora il limite
                "duration": round(float(min(n.end, limit) - n.start), 4),
            })
    events.sort(key=lambda e: e["time"])
    return {
        "duration": round(limit, 4),
        "full_duration": round(full_duration, 4),
        "truncated": limit < full_duration,
        "num_events": len(events),
        "events": events,
    }


# ---------- sintesi audio di lavoro (con soft-limiter) ----------
def synth_audio(events, duration):
    # la coda arriva esattamente alla fine dell'ultima nota (niente secondo extra
    # che farebbe suonare l'audio oltre il brano vero)
    if events:
        last_end = max(e["time"] + e["duration"] for e in events)
    else:
        last_end = duration
    total = int(min(duration, last_end) * SR) + 1
    mix = np.zeros(total)
    for ev in events:
        freq, dur = ev["freq"], ev["duration"]
        n = int(dur * SR)
        if n <= 0:
            continue
        t = np.arange(n) / SR
        wave = np.sin(2 * np.pi * freq * t) + 0.15 * np.sin(2 * np.pi * 2 * freq * t)
        atk = min(int(0.015 * SR), n // 2)
        rel = min(int(0.25 * SR), n - atk)
        env = np.ones(n)
        if atk > 0:
            env[:atk] = np.linspace(0, 1, atk)
        if rel > 0:
            env[-rel:] = np.cos(np.linspace(0, np.pi / 2, rel))
        i0 = int(ev["time"] * SR)
        if i0 >= total:
            continue
        seg = wave * env * ev["amplitude"] * 0.25
        end = min(i0 + len(seg), total)
        mix[i0:end] += seg[:end - i0]
    mix = np.tanh(mix * 0.6) / 0.6
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix / peak * 0.9
    return mix.astype("float32")


# ---------- il componente di visualizzazione (p5.js) ----------
def build_viz_html(score_json, audio_b64, mask_key="seoul"):
    mask = ALL_MASKS.get(mask_key, ALL_MASKS["seoul"])
    MASK_B64 = mask["b64"]
    mask_w, mask_h = mask["w"], mask["h"]

    # ITALIA: la maschera resta DRITTA e il canvas ha le sue proporzioni normali.
    # La rotazione avviene DENTRO p5 al momento del disegno (non con CSS, non
    # ruotando l'immagine): le curve si generano con la logica standard su uno
    # "spazio sdraiato" e vengono appoggiate ruotate sul canvas dritto.
    rotate_draw = (mask_key == "italy")
    rotate_js = "true" if rotate_draw else "false"

    aspect = mask_w / mask_h
    if aspect >= 1:
        cw = 1200; ch = int(1200 / aspect)
    else:
        ch = 1000; cw = int(1000 * aspect)
    """Costruisce la visualizzazione p5.js dentro un IFRAME isolato."""
    canvas_css = "max-width:100%;max-height:100%;height:auto;width:auto;"
    inner = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
html,body{{margin:0;height:100%;background:#0a0d12;font-family:-apple-system,system-ui,sans-serif;overflow:hidden;}}
#wrap{{display:flex;flex-direction:column;height:100vh;}}
.ctrl{{flex:0 0 auto;text-align:center;padding:8px;background:#0a0d12;}}
#viz-holder{{flex:1 1 auto;display:flex;justify-content:center;align-items:center;min-height:0;overflow:hidden;}}
#viz-holder canvas{{{canvas_css}}}
button{{padding:7px 16px;margin:3px;border-radius:6px;border:1px solid #444;background:#1a2028;color:#eee;cursor:pointer;font-size:14px;}}
button:hover{{background:#232b35;}}
.bar-wrap{{display:flex;align-items:center;gap:10px;max-width:600px;margin:6px auto 0;padding:0 12px;}}
#seek{{flex:1;height:5px;-webkit-appearance:none;appearance:none;background:#2a3340;border-radius:3px;outline:none;cursor:pointer;}}
#seek::-webkit-slider-thumb{{-webkit-appearance:none;width:13px;height:13px;border-radius:50%;background:#6da8ff;cursor:pointer;}}
#seek::-moz-range-thumb{{width:13px;height:13px;border-radius:50%;background:#6da8ff;border:none;cursor:pointer;}}
#tlabel{{color:#aaa;font-size:12px;font-variant-numeric:tabular-nums;min-width:82px;}}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.9.4/p5.min.js"></script>
</head><body>
<div id="wrap">
<div class="ctrl">
  <button onclick="window.vizPlay()">&#9654; Play</button>
  <button onclick="window.vizPause()">&#9208; Pause</button>
  <button onclick="window.vizFull()">&#9974; Fullscreen</button>
  <button onclick="window.vizSaveImg()">&#128247; Save image</button>
  <button id="recBtn" onclick="window.vizRec()">&#9210; Record video</button>
  <div class="bar-wrap">
    <input type="range" id="seek" min="0" max="1000" value="0">
    <span id="tlabel">0:00 / 0:00</span>
  </div>
  <div id="dbg" style="color:#5a6b7a;font-size:11px;font-variant-numeric:tabular-nums;margin-top:4px;"></div>
</div>
<div id="viz-holder"></div>
<audio id="viz-player" style="display:none;"></audio>
</div>
<script>
(function() {{
  const MASK_B64 = "{MASK_B64}";
  const SCORE = {json.dumps(score_json)};
  const AUDIO_B64 = "{audio_b64}";
  const ROTATE_DRAW = {rotate_js};  // Italia: disegno sdraiato, composizione ruotata

  let W = {cw}, H = {ch}, BIG_MULT = 2;
  // dimensioni dello spazio di DISEGNO: se ruotato, sono scambiate (sdraiate)
  let DW = ROTATE_DRAW ? H : W;
  let DH = ROTATE_DRAW ? W : H;
  let events = SCORE.events.slice().sort((a,b)=>a.time-b.time);
  let sortedMidis = events.filter(e=>e.freq!=null).map(e=>69+12*Math.log2(e.freq/440)).sort((a,b)=>a-b);
  let nextIdx = 0, lastT = 0;
  let accum, flashG, maskCanvas, maskReady=false, player, useMask=true;
  let flashes = [], FLASH_DUR=0.5;

  function midiPct(m){{ let a=sortedMidis; if(!a.length)return .5; let lo=0,hi=a.length;
    while(lo<hi){{let mid=(lo+hi)>>1; if(a[mid]<=m)lo=mid+1; else hi=mid;}} return lo/a.length; }}

  function hslRgb(h,s,l){{ h/=360; let r,g,b; if(s==0){{r=g=b=l;}} else {{
    let f=(p,q,t)=>{{if(t<0)t+=1;if(t>1)t-=1;if(t<1/6)return p+(q-p)*6*t;if(t<1/2)return q;
      if(t<2/3)return p+(q-p)*(2/3-t)*6;return p;}};
    let q=l<.5?l*(1+s):l+s-l*s,p=2*l-q; r=f(p,q,h+1/3);g=f(p,q,h);b=f(p,q,h-1/3);}}
    return [r*255,g*255,b*255]; }}

  new p5(function(p){{
    p.setup = function(){{
      let c = p.createCanvas(W,H); c.parent('viz-holder');
      p.frameRate(30);  // 30fps: piu' leggero e compatibile, resa visiva quasi identica
      window._p5canvas = c.elt;  // riferimento al canvas per i pulsanti esterni
      window._p5inst = p;
      accum = p.createGraphics(DW,DH); accum.clear();
      flashG = p.createGraphics(DW,DH); flashG.clear();
      maskCanvas = document.createElement('canvas'); maskCanvas.width=W; maskCanvas.height=H;
      let mi = new Image();
      mi.onload = ()=>{{ maskCanvas.getContext('2d').drawImage(mi,0,0,W,H); maskReady=true; }};
      mi.src = MASK_B64;
      player = document.getElementById('viz-player');
      // Converto i dati audio in un BLOB URL invece di usare una data URL gigante.
      // I browser non bufferizzano bene gli audio grandi incorporati come data URL
      // (possono riavviarsi a meta' riproduzione); un blob URL viene gestito come
      // un file audio normale, con buffering corretto.
      (function(){{
        let bin = atob(AUDIO_B64);
        let bytes = new Uint8Array(bin.length);
        for(let i=0;i<bin.length;i++) bytes[i] = bin.charCodeAt(i);
        let blob = new Blob([bytes], {{type:'audio/wav'}});
        player.src = URL.createObjectURL(blob);
      }})();
      player.load();

      // barra di avanzamento + timer
      let seek = document.getElementById('seek');
      let tlabel = document.getElementById('tlabel');
      let dragging = false;
      function fmt(s){{ s=Math.max(0,s|0); return (s/60|0)+':'+('0'+(s%60)).slice(-2); }}

      // l'audio aggiorna la barra (solo quando NON la stai trascinando)
      player.addEventListener('timeupdate', ()=>{{
        if(dragging || !player.duration) return;
        seek.value = (player.currentTime/player.duration*1000)|0;
        tlabel.textContent = fmt(player.currentTime)+' / '+fmt(player.duration);
      }});
      player.addEventListener('loadedmetadata', ()=>{{
        tlabel.textContent = '0:00 / '+fmt(player.duration);
      }});

      // la barra muove l'audio SOLO durante un trascinamento reale del mouse.
      // Uso pointerdown/pointerup: nessun evento automatico puo' spostare l'audio.
      seek.addEventListener('pointerdown', ()=>{{ dragging = true; }});
      seek.addEventListener('pointerup', ()=>{{
        if(player.duration) player.currentTime = seek.value/1000*player.duration;
        dragging = false;
      }});
      // mentre trascini, aggiorno solo l'etichetta (non tocco l'audio)
      seek.addEventListener('input', ()=>{{
        if(dragging && player.duration)
          tlabel.textContent = fmt(seek.value/1000*player.duration)+' / '+fmt(player.duration);
      }});

      // FINE BRANO: stop netto, nessun riavvio
      player.loop = false;
      player.addEventListener('ended', ()=>{{ player.pause(); }});

      p.background(10,13,18);
    }};

    p.draw = function(){{
      p.background(10,13,18);
      if(player){{
        let ct = player.currentTime;
        // se il tempo e' andato indietro rispetto agli eventi gia' disegnati
        // (barra trascinata indietro, restart), ricostruisco l'accumulo da zero
        // fino al punto attuale. Cosi' il canvas non si congela mai.
        if(nextIdx>0 && ct < (events[nextIdx-1] ? events[nextIdx-1].time : 0)){{
          accum.clear(); flashes = []; nextIdx = 0;
        }}
        if(!player.paused){{
          // limito quanti eventi disegno per frame: se sono molto indietro
          // (dopo un salto), evito di bloccare il browser disegnandone centinaia
          // in un colpo solo. Ne disegno un blocco per frame finche' recupero.
          let budget = 120;
          while(nextIdx<events.length && events[nextIdx].time<=ct && budget-->0){{
            drawEvent(events[nextIdx]); nextIdx++;
          }}
          lastT = ct;
        }}
        let dbg = document.getElementById('dbg');
        if(dbg) dbg.textContent = 'audio t='+ct.toFixed(1)+'s / dur='+(player.duration||0).toFixed(1)
          +'s | eventi '+nextIdx+'/'+events.length;
      }}
      updateFlashes(p);
      // compongo accumulo+flash sul canvas. Se ROTATE_DRAW, ruoto di -90° cosi'
      // lo spazio di disegno sdraiato (DWxDH) si appoggia dritto sul canvas (WxH):
      // cio' che era a destra (acuto) va in alto, cio' che era a sinistra (grave)
      // va in basso. La maschera (dritta) si applica DOPO, senza rotazione.
      p.push();
      if(ROTATE_DRAW){{
        p.translate(W/2, H/2);
        p.rotate(-p.HALF_PI);
        p.imageMode(p.CENTER);
        p.image(accum, 0, 0); p.image(flashG, 0, 0);
        p.imageMode(p.CORNER);
      }} else {{
        p.image(accum,0,0); p.image(flashG,0,0);
      }}
      p.pop();
      if(useMask && maskReady){{
        p.drawingContext.save();
        p.drawingContext.globalCompositeOperation='destination-in';
        p.drawingContext.drawImage(maskCanvas,0,0);
        p.drawingContext.globalCompositeOperation='source-over';
        p.drawingContext.restore();
      }}
    }};

    p.keyPressed = function(){{ if(p.key=='m'||p.key=='M') useMask=!useMask; }};

    function drawEvent(ev){{
      if(ev.freq==null) return;
      let midi=69+12*Math.log2(ev.freq/440);
      let px=midiPct(midi);
      let x=p.lerp(DW*0.08,DW*0.92,px);
      let y=DH*(0.2+0.6*p.noise(nextIdx*0.1));
      let amp=(typeof ev.amplitude=='number'&&isFinite(ev.amplitude))?ev.amplitude:0.5;
      let len=p.map(amp,0,1,DH*0.15,DH*0.45);
      let dir=p.noise(x*0.004,y*0.004)*p.TWO_PI;
      let x2=x+p.cos(dir)*len, y2=y+p.sin(dir)*len;
      let ad=p.map(amp,0,1,DH*0.38,DH*1.5);
      let a1=p.noise(x*0.005+10)*p.TWO_PI, a2=p.noise(x2*0.005+20)*p.TWO_PI;
      let a1x=x+p.cos(a1)*ad,a1y=y+p.sin(a1)*ad,a2x=x2+p.cos(a2)*ad,a2y=y2+p.sin(a2)*ad;
      let hue=p.lerp(220,140,px); let [r,g,b]=hslRgb(hue,0.72,0.62);
      accum.stroke(r,g,b,26); accum.strokeWeight(1.2); accum.noFill();
      accum.curve(a1x,a1y,x,y,x2,y2,a2x,a2y);
      flashes.push({{x,y,x2,y2,a1x,a1y,a2x,a2y,r,g,b,born:p.millis()}});
    }}

    function updateFlashes(p){{
      flashG.clear();
      for(let i=flashes.length-1;i>=0;i--){{
        let f=flashes[i]; let el=(p.millis()-f.born)/1000;
        if(el>FLASH_DUR){{flashes.splice(i,1);continue;}}
        let t=el/FLASH_DUR, a=p.lerp(180,0,t), w=p.lerp(2.2,0.6,t);
        flashG.stroke(f.r,f.g,f.b,a); flashG.strokeWeight(w); flashG.noFill();
        flashG.curve(f.a1x,f.a1y,f.x,f.y,f.x2,f.y2,f.a2x,f.a2y);
      }}
    }}
  }});

  window.vizPlay = function(){{ if(player){{ player.play(); }} }};
  window.vizPause = function(){{ if(player){{ player.pause(); }} }};

  // schermo intero sul contenitore della visualizzazione
  window.vizFull = function(){{
    let el = document.getElementById('wrap');
    if(document.fullscreenElement){{ document.exitFullscreen(); }}
    else if(el.requestFullscreen){{ el.requestFullscreen(); }}
    else if(el.webkitRequestFullscreen){{ el.webkitRequestFullscreen(); }}
  }};

  // salva il fotogramma corrente come PNG
  window.vizSaveImg = function(){{
    if(!window._p5canvas) return;
    let a = document.createElement('a');
    a.download = 'seoul_quadro.png';
    a.href = window._p5canvas.toDataURL('image/png');
    a.click();
  }};

  // registra un video del canvas (con audio) mentre suona, dal vivo
  let mediaRec=null, recChunks=[], recording=false;
  window.vizRec = function(){{
    let btn = document.getElementById('recBtn');
    if(!recording){{
      if(!window._p5canvas) return;
      recChunks=[];
      let vstream = window._p5canvas.captureStream(30);
      // provo ad agganciare anche l'audio dell'elemento player
      try{{
        let actx = new (window.AudioContext||window.webkitAudioContext)();
        let src = actx.createMediaElementSource(player);
        let dst = actx.createMediaStreamDestination();
        src.connect(dst); src.connect(actx.destination);
        dst.stream.getAudioTracks().forEach(t=>vstream.addTrack(t));
      }}catch(e){{ /* se l'audio non si aggancia, registro solo video */ }}
      mediaRec = new MediaRecorder(vstream, {{mimeType:'video/webm'}});
      mediaRec.ondataavailable = e=>{{ if(e.data.size>0) recChunks.push(e.data); }};
      mediaRec.onstop = ()=>{{
        let blob = new Blob(recChunks, {{type:'video/webm'}});
        let a = document.createElement('a');
        a.download = 'seoul_video.webm';
        a.href = URL.createObjectURL(blob); a.click();
      }};
      mediaRec.start();
      recording=true; btn.textContent='\\u25A0 Stop & download';
      // faccio ripartire il brano da capo cosi' il video parte dall'inizio
      if(player){{ player.currentTime=0; player.play(); }}
    }} else {{
      mediaRec.stop(); recording=false;
      btn.innerHTML='&#9210; registra video';
    }}
  }};

  window.addEventListener('keydown', function(e){{ if(e.key=='m'||e.key=='M'){{ useMask=!useMask; }} }});
}})();
</script>
</body></html>"""

    # incapsulo in un iframe via srcdoc (base64 per evitare problemi di escape)
    inner_b64 = base64.b64encode(inner.encode("utf-8")).decode()
    return (f'<iframe src="data:text/html;base64,{inner_b64}" '
            f'style="width:100%;height:820px;border:none;border-radius:8px;" '
            f'allow="fullscreen" allowfullscreen '
            f'sandbox="allow-scripts allow-same-origin allow-downloads"></iframe>')


# ---------- funzione principale chiamata da Gradio ----------
def process(midi_file, audio_file, max_minutes, mask_choice):
    if midi_file is None:
        return "Upload a MIDI file to start.", None, None, ""

    # dalla label leggibile alla chiave interna
    mask_key = {v: k for k, v in MASK_LABELS.items()}.get(mask_choice, "seoul")

    max_seconds = float(max_minutes) * 60 if max_minutes else None
    score = extract_score(midi_file, max_seconds=max_seconds)

    # audio: quello alternativo se fornito, altrimenti sintetizzato
    if audio_file is not None:
        audio_data, sr = sf.read(audio_file)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        # taglio l'audio al limite di durata scelto (non tocco i tempi degli eventi:
        # audio e MIDI arrivano gia' allineati dall'utente)
        if max_seconds is not None:
            audio_data = audio_data[:int(max_seconds * sr)]
        buf = io.BytesIO()
        sf.write(buf, audio_data, sr, format="WAV")
        audio_bytes = buf.getvalue()
        audio_out_path = _base_name(midi_file, mask_choice, "_audio.wav")
        sf.write(audio_out_path, audio_data, sr)
    else:
        mix = synth_audio(score["events"], score["duration"])
        buf = io.BytesIO()
        sf.write(buf, mix, SR, format="WAV")
        audio_bytes = buf.getvalue()
        audio_out_path = _base_name(midi_file, mask_choice, "_audio.wav")
        sf.write(audio_out_path, mix, SR)

    audio_b64 = base64.b64encode(audio_bytes).decode()

    # salvo il JSON scaricabile con nome derivato dal MIDI
    json_path = _base_name(midi_file, mask_choice, "_score.json")
    with open(json_path, "w") as jf:
        json.dump(score, jf)

    viz = build_viz_html(score, audio_b64, mask_key)
    note = ""
    if score.get("truncated"):
        note = (f" · trimmed to the first {score['duration']:.0f}s "
                f"(original {score['full_duration']:.0f}s)")
    info = f"✓ {score['num_events']} notes · {score['duration']:.0f}s{note}"
    return viz, json_path, audio_out_path, info


# ---------- interface ----------
def _base_name(midi_file, mask_choice, suffix):
    """Nome file di output derivato dal MIDI caricato + forma scelta."""
    import re
    raw = os.path.splitext(os.path.basename(midi_file))[0] if midi_file else "soundmap"
    raw = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_") or "soundmap"
    shape = mask_choice.lower().replace(" ", "-") if mask_choice else "map"
    d = tempfile.mkdtemp()
    return os.path.join(d, f"{raw}_{shape}{suffix}")


def process_mp4(midi_file, audio_file, max_minutes, mask_choice, progress=gr.Progress()):
    """Genera un video MP4 lato Python (rendering offline, qualita' piena)."""
    import render_mp4
    if midi_file is None:
        return None, "Upload a MIDI file to start."

    mask_key = {v: k for k, v in MASK_LABELS.items()}.get(mask_choice, "seoul")
    max_seconds = float(max_minutes) * 60 if max_minutes else None
    score = extract_score(midi_file, max_seconds=max_seconds)

    # audio: alternativo o sintetizzato
    if audio_file is not None:
        audio_data, sr = sf.read(audio_file)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        if max_seconds is not None:
            audio_data = audio_data[:int(max_seconds * sr)]
        atmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(atmp.name, audio_data, sr)
        audio_path = atmp.name
    else:
        mix = synth_audio(score["events"], score["duration"])
        atmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(atmp.name, mix, SR)
        audio_path = atmp.name

    mask_b64 = ALL_MASKS.get(mask_key, ALL_MASKS["seoul"])["b64"]
    out_path = _base_name(midi_file, mask_choice, ".mp4")

    def prog(p):
        progress(p, desc="Rendering video…")

    render_mp4.render(score, audio_path, mask_b64, mask_key, out_path,
                      max_seconds=max_seconds, progress=prog)
    dur = min(score["duration"], max_seconds) if max_seconds else score["duration"]
    return out_path, f"✓ MP4 ready · {score['num_events']} notes · {dur:.0f}s"


def on_browser(midi_file, audio_file, max_minutes, mask_choice):
    """Pulsante 'View in browser': mostra l'anteprima interattiva."""
    viz, json_p, audio_p, info = process(midi_file, audio_file, max_minutes, mask_choice)
    return (gr.update(value=viz, visible=True),        # viz_out
            gr.update(value=json_p, visible=True),     # json_out
            gr.update(value=audio_p, visible=True),    # audio_out
            info,                                       # status
            gr.update(value=None, visible=False))      # mp4_out nascosto


def on_mp4(midi_file, audio_file, max_minutes, mask_choice, progress=gr.Progress()):
    """Pulsante 'Generate MP4': rendering offline e file scaricabile."""
    mp4_path, msg = process_mp4(midi_file, audio_file, max_minutes, mask_choice, progress)
    return (gr.update(value="", visible=False),        # viz_out nascosto
            gr.update(value=None, visible=False),      # json_out nascosto
            gr.update(value=None, visible=False),      # audio_out nascosto
            msg,                                        # status
            gr.update(value=mp4_path, visible=True))   # mp4_out visibile


# ---------- interface ----------
_css = """
/* --- tema scuro: fondo e pannelli scuri, solo i CONTROLLI restano chiari --- */
.gradio-container { background:#0a0d12 !important;
  font-family:-apple-system,'Inter',system-ui,sans-serif !important; }
#app-title { font-size:30px; font-weight:650; letter-spacing:-0.5px;
  color:#ffffff; margin-bottom:2px; }
#app-sub { color:#c3ccd8 !important; font-size:14px; line-height:1.5; }
#out-note, #out-note * { color:#c3ccd8 !important; }
/* pannelli/blocchi e aree di upload: SCURI come lo sfondo */
.gradio-container .block, .gradio-container .form,
.gradio-container [class*="panel"], .gradio-container [class*="upload"],
.gradio-container [data-testid*="upload"] {
  background:#12161d !important; border:1px solid #232b35 !important;
  border-radius:12px !important; }
/* testo dentro le aree di upload (Drop file, or, Click) leggibile su scuro */
.gradio-container [class*="upload"] *,
.gradio-container [data-testid*="upload"] * { color:#c3ccd8 !important; }
/* etichette dei campi: CHIARE (stanno su pannello scuro) */
.gradio-container label span, .gradio-container .block-title,
.gradio-container [class*="block-title"], .gradio-container label > span:first-child {
  color:#dbe2ea !important; font-weight:600 !important; }
/* CONTROLLI chiari: dropdown, campo numero, slider-track */
.gradio-container [class*="dropdown"] input,
.gradio-container [class*="dropdown"] [class*="single-select"],
.gradio-container input[type="number"], .gradio-container input[type="text"] {
  background:#eef1f5 !important; color:#1a2029 !important; border-color:#c9d2dd !important; }
.gradio-container ul[role="listbox"] { background:#eef1f5 !important; color:#1a2029 !important; }
.gradio-container li[role="option"] { color:#1a2029 !important; }
.gradio-container li[role="option"]:hover,
.gradio-container li[role="option"][aria-selected="true"] {
  background:#d7dee7 !important; }
/* freccia dropdown: solo icona, niente riquadro nero */
.gradio-container [class*="dropdown"] [class*="icon"],
.gradio-container [class*="dropdown"] svg {
  background:transparent !important; color:#4a5563 !important; }
.gradio-container [class*="dropdown"] button,
.gradio-container [class*="dropdown"] [class*="token"] {
  background:transparent !important; border:none !important; }
/* i due pulsanti modalita': chiari, testo scuro, hover verde */
.gradio-container #btn-browser, .gradio-container #btn-mp4 {
  background:#eef1f5 !important; color:#1a2029 !important;
  border:1px solid #c9d2dd !important; font-weight:600 !important; }
.gradio-container #btn-browser:hover, .gradio-container #btn-mp4:hover {
  background:#e2e7ee !important; border-color:#3fb58f !important; }
#downloads { gap:8px; }
"""


with gr.Blocks(title="Sound Map", theme=gr.themes.Base(), css=_css) as demo:
    gr.HTML('<div id="app-title">Sound Map</div>'
            '<div id="app-sub">A piece of music becomes a drawing over a map. '
            'Upload a MIDI file: each note traces a curve, placed by pitch. '
            'Choose a shape, and use the synthesized audio or upload your own.</div>')
    with gr.Row():
        with gr.Column(scale=1):
            midi_in = gr.File(label="MIDI file (.mid)", file_types=[".mid", ".midi"])
            audio_in = gr.File(label="Alternative audio (optional)", file_types=["audio"])
            mask_choice = gr.Dropdown(
                choices=MENU_CHOICES,
                value=MASK_LABELS.get("seoul", MENU_CHOICES[0]),
                label="Shape / map",
                filterable=True,
                info="Type to search — countries and major Korean cities")
            max_min = gr.Slider(minimum=1, maximum=30, value=10, step=1,
                                label="Maximum duration (minutes)",
                                info="longer pieces are trimmed to keep things light")
            gr.Markdown("**Output** — Browser: live interactive preview. "
                        "MP4: full-quality rendered file (slower). "
                        "Different renderers, so results are not pixel-identical.",
                        elem_id="out-note")
            with gr.Row():
                btn_browser = gr.Button("▶ View in browser", elem_id="btn-browser")
                btn_mp4 = gr.Button("● Generate MP4", elem_id="btn-mp4")
        with gr.Column(scale=2):
            status = gr.Markdown("")
            with gr.Row(elem_id="downloads"):
                mp4_out = gr.File(label="Download MP4", visible=False)
                json_out = gr.File(label="Download score (JSON)", visible=False)
                audio_out = gr.File(label="Download audio", visible=False)
            viz_out = gr.HTML(label="Preview")

    outs = [viz_out, json_out, audio_out, status, mp4_out]

    def _clear_before_browser():
        return (gr.update(value="", visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                "Loading…",
                gr.update(value=None, visible=False))
    btn_browser.click(_clear_before_browser, inputs=None, outputs=outs).then(
        on_browser, inputs=[midi_in, audio_in, max_min, mask_choice], outputs=outs)

    # MP4: prima nascondo l'anteprima e i download (evita barre sovrapposte),
    # poi lancio il rendering. Due step concatenati.
    def _clear_before_mp4():
        return (gr.update(value="", visible=False),   # viz_out via
                gr.update(value=None, visible=False),  # json_out via
                gr.update(value=None, visible=False),  # audio_out via
                "Rendering…",                          # status
                gr.update(value=None, visible=False))  # mp4_out via
    btn_mp4.click(_clear_before_mp4, inputs=None, outputs=outs).then(
        on_mp4, inputs=[midi_in, audio_in, max_min, mask_choice], outputs=outs)

if __name__ == "__main__":
    demo.launch()
