const $ = id => document.getElementById(id);
const video = $('video'), overlay = $('overlay'), capture = document.createElement('canvas');
let stream = null, timer = null, active = false, started = 0, previousState = '', audio = null, lastBeep = 0;
let generation = 0, busy = false;
let previousYawn = false;
let previousHeadCue = false;
let previousQnxState = '';
let previousEmergency = false;
const session = crypto.randomUUID();
const labels = {idle:'Ready when you are', no_face:'No face in view', multiple_faces:'One driver at a time',
  model_missing:'Models not ready', warming_up:'Building fatigue baseline', alert:'Alert',
  low_vigilance:'Low vigilance', checking:'Checking drowsiness', drowsy:'Time for a break', uncertain:'Uncertain', error:'Connection interrupted'};

function event(text) {
  const list = $('events');
  if (list.querySelector('.muted')) list.textContent = '';
  const row = document.createElement('li'), time = document.createElement('time'), message = document.createElement('span');
  time.textContent = new Date().toLocaleTimeString(); message.textContent = text;
  row.append(time, message); list.prepend(row);
  while (list.children.length > 30) list.lastChild.remove();
}
function modelStatus(data) {
  const models = data?.models || {};
  for (const task of ['eye_state','yawn']) {
    const item = models[task] || {loaded:false,error:'Model status unavailable'};
    const element = $(task==='eye_state'?'eyeStateModel':'yawnModel');
    if (element) {
      element.textContent = item.loaded ? `Loaded · epoch ${item.epoch}` : 'Not loaded';
      element.title = item.error || '';
    }
  }
  $('modelNotice').hidden = ['eye_state','yawn'].every(task => models[task]?.loaded);
  showQnx(data?.qnx || {enabled:false,connected:false});
}
function showQnx(qnx) {
  const link = !qnx.enabled ? 'Disabled' : qnx.connected ? 'Connected' : 'Waiting';
  $('qnxLink').textContent = link;
  $('qnxDecision').textContent = (qnx.decision_state || 'unavailable').replaceAll('_',' ');
  $('qnxLatency').textContent = qnx.decision_us == null ? '— µs' : `${qnx.decision_us} µs`;
  $('qnxAge').textContent = qnx.packet_age_ms == null ? '— ms' : `${qnx.packet_age_ms} ms`;
  $('qnxRto').textContent = qnx.recovery_rto_ms == null ? '— ms' : `${qnx.recovery_rto_ms} ms`;
  $('vehicleSpeed').textContent = qnx.simulated_speed_kph == null ? '— km/h' : `${Number(qnx.simulated_speed_kph).toFixed(0)} km/h`;
  $('emergencyState').textContent = qnx.emergency_active ? 'ACTIVE' : 'Standby';
  $('emergencyState').classList.toggle('emergency-active', Boolean(qnx.emergency_active));
  if (qnx.emergency_active && !previousEmergency) event('QNX prolonged-drowsiness emergency activated');
  if (!qnx.emergency_active && previousEmergency) event('QNX emergency cleared');
  previousEmergency = Boolean(qnx.emergency_active);
  const current = `${link}:${qnx.decision_state || ''}`;
  if (qnx.enabled && current !== previousQnxState) event(`QNX ${link.toLowerCase()} · ${qnx.decision_state || 'no decision'}`);
  previousQnxState = current;
}
function clearScores() {
  for (const id of ['alert','low','drowsy']) { $(id+'Value').textContent = '—'; $(id+'Bar').style.width = '0%'; }
  $('yawnValue').textContent = '—';
}
function showHead(head, faceCount=0) {
  head ||= {};
  const calibrated = Boolean(head.calibrated), cue = Boolean(head.cue);
  const progress = Math.max(0, Math.min(100, Number(head.calibration_progress || 0)));
  $('headStatus').textContent = cue ? `WARNING · ${head.status || 'movement'}` :
    calibrated ? (head.status || 'stable') : faceCount===1 ? `Calibrating ${progress}%` : 'Waiting';
  $('headCue').textContent = cue ? 'MOVEMENT DETECTED' : calibrated ? 'MONITORING' : `CALIBRATING ${progress}%`;
  $('headPanel').classList.toggle('cue', cue);
  $('headYaw').textContent = calibrated ? Number(head.yaw || 0).toFixed(3) : '—';
  $('headPitch').textContent = calibrated ? Number(head.pitch || 0).toFixed(3) : '—';
  $('headRoll').textContent = calibrated ? `${Number(head.roll || 0).toFixed(1)}°` : '—';
  $('headStatus').title = calibrated ? `Yaw ${head.yaw}, pitch ${head.pitch}, roll ${head.roll}°` : '';
}
function state(name, message) {
  $('statusCard').dataset.state = name; $('stateTitle').textContent = labels[name] || name;
  $('stateMessage').textContent = message; $('stateSymbol').textContent = name==='alert'?'✓':name==='drowsy'?'!':'—';
  if (name !== previousState) { event(labels[name] || name); previousState = name; }
}
function beep() {
  if (!$('sound').checked || !audio || performance.now()-lastBeep<5000) return;
  lastBeep = performance.now();
  const tone = audio.createOscillator(), gain = audio.createGain();
  tone.frequency.value = 880; gain.gain.value = .12; tone.connect(gain); gain.connect(audio.destination);
  tone.start(); tone.stop(audio.currentTime+.35);
}
function show(data) {
  data ||= {};
  const predictions = data.predictions || {};
  let message = data.message || 'Waiting for valid inference output';
  if (data.state === 'warming_up' && data.required_frames) {
    const collected = data.sampled_frames || 0;
    const remaining = Math.max(0, data.required_frames - collected);
    message = `Collecting ${data.required_frames} one-second samples: ${collected}/${data.required_frames}. About ${remaining} seconds remaining.`;
  }
  modelStatus(data); state(data.state || 'uncertain', message); clearScores();
  showQnx(data.qnx || {enabled:false,connected:false});
  const probabilities = predictions.drowsiness;
  if (probabilities) for (const [key,id] of [['alert','alert'],['low_vigilance','low'],['drowsy','drowsy']]) {
    const value = (probabilities[key]*100).toFixed(0)+'%'; $(id+'Value').textContent = value; $(id+'Bar').style.width = value;
  }
  if (predictions.yawn) $('yawnValue').textContent = (predictions.yawn.yawn*100).toFixed(0)+'%'+(data.yawn_detected?' · detected':'');
  if (data.yawn_detected && !previousYawn) event('Yawn evidence detected');
  previousYawn = Boolean(data.yawn_detected);
  $('latency').textContent = data.inference_ms == null ? '— ms' : `${data.inference_ms} ms`;
  $('faceStatus').textContent = data.face_count===1?'Tracked':data.face_count>1?'Multiple faces':'No face';
  showHead(data.head, data.face_count);
  if (data.head?.cue && !previousHeadCue) event(`Sustained head movement: ${data.head.status}`);
  previousHeadCue = Boolean(data.head?.cue);
  $('windowCount').textContent = `${data.sampled_frames || 0} / ${data.required_frames || '—'}`;
  $('windowBar').style.width = (data.required_frames ? (data.sampled_frames || 0)/data.required_frames*100 : 0)+'%';
  $('feedTag').textContent = (labels[data.state] || data.state).toUpperCase();
  const ctx = overlay.getContext('2d'); ctx.clearRect(0,0,overlay.width,overlay.height);
  if (data.boxes) {
    ctx.lineWidth = 2; ctx.strokeStyle = data.alert?'#ff827d':'#65dcc5';
    for (const key of ['face','mouth']) { const [x1,y1,x2,y2] = data.boxes[key]; ctx.strokeRect(x1,y1,x2-x1,y2-y1); }
    for (const [x1,y1,x2,y2] of data.boxes.eyes || []) ctx.strokeRect(x1,y1,x2-x1,y2-y1);
  }
  if (data.alert) beep();
}
async function tick(run) {
  if (!active || run !== generation) return;
  if (busy) { timer=setTimeout(()=>tick(run),100); return; }
  if (video.readyState<2) { timer=setTimeout(()=>tick(run),200); return; }
  busy = true;
  try {
    const scale = Math.min(1,640/video.videoWidth); capture.width = Math.round(video.videoWidth*scale); capture.height = Math.round(video.videoHeight*scale);
    overlay.width=capture.width; overlay.height=capture.height;
    capture.getContext('2d').drawImage(video,0,0,capture.width,capture.height);
    const blob = await new Promise(resolve=>capture.toBlob(resolve,'image/jpeg',.8));
    const result=await fetch('/api/detect',{method:'POST',headers:{'Content-Type':'image/jpeg','X-Session-ID':session},body:blob,signal:AbortSignal.timeout(10000)});
    const data=await result.json(); if(!result.ok) throw Error(data.error);
    if (active && run===generation) show(data);
  } catch(error) {
    if(active && run===generation) {
      clearScores(); state('error',error.message); $('faceStatus').textContent='Unavailable';
      overlay.getContext('2d').clearRect(0,0,overlay.width,overlay.height);
      $('feedTag').textContent='CONNECTION INTERRUPTED'; $('windowBar').style.width='0%';
      $('windowCount').textContent='—'; $('latency').textContent='— ms'; previousYawn=false; previousHeadCue=false;
      showHead({}, 0);
    }
  } finally { busy=false; }
  if(active && run===generation) timer=setTimeout(()=>tick(run),500);
}
$('start').onclick=async()=>{
  $('start').disabled=true;
  try {
    stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},facingMode:'user'},audio:false});
    video.srcObject=stream; await video.play();
    // Keep overlay coordinates aligned with the camera's aspect ratio.
    video.parentElement.style.aspectRatio=`${video.videoWidth}/${video.videoHeight}`;
    video.parentElement.classList.add('running'); $('empty').hidden=true;
    active=true; started=Date.now(); generation++; $('stop').disabled=false;
    $('cameraLabel').textContent='CONNECTED'; $('liveDot').classList.add('active');
    $('cameraHelp').textContent='Monitoring locally. Keep one face in view; pause with Stop.';
    audio ||= new (window.AudioContext||window.webkitAudioContext)(); await audio.resume();
    await fetch('/api/reset',{method:'POST'}); tick(generation);
  } catch(error) { stop(); $('cameraHelp').textContent=`Camera unavailable: ${error.message}. Check browser and Windows camera permissions.`; }
};
function stop() {
  active=false; previousYawn=false; previousHeadCue=false; generation++; clearTimeout(timer); stream?.getTracks().forEach(track=>track.stop()); stream=null;
  video.srcObject=null; video.parentElement.classList.remove('running'); $('empty').hidden=false;
  $('start').disabled=false; $('stop').disabled=true; $('liveDot').classList.remove('active');
  $('cameraLabel').textContent='CAMERA OFF'; $('feedTag').textContent='WAITING FOR CAMERA';
  overlay.getContext('2d').clearRect(0,0,overlay.width,overlay.height); clearScores(); state('idle','Camera stopped.');
  $('windowCount').textContent='0 / 8'; $('windowBar').style.width='0%'; $('faceStatus').textContent='Waiting'; $('headStatus').textContent='Waiting'; $('latency').textContent='— ms';
  showHead({}, 0);
  fetch('/api/reset',{method:'POST'}).catch(()=>{});
}
$('stop').onclick=stop;
$('reload').onclick=async()=>{
  try { const result=await fetch('/api/reload',{method:'POST'}); if(!result.ok) throw Error('Reload failed'); modelStatus(await result.json()); clearScores(); event('Model files reloaded'); }
  catch(error){event(error.message);}
};
$('clearEvents').onclick=()=>{$('events').textContent='';};
setInterval(()=>{if(active){const seconds=Math.floor((Date.now()-started)/1000);$('sessionTime').textContent=`SESSION ${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')}`;}},1000);
window.addEventListener('pagehide',()=>{stream?.getTracks().forEach(track=>track.stop());});
fetch('/api/status').then(r=>{if(!r.ok)throw Error('Server unavailable');return r.json();}).then(modelStatus).catch(error=>state('error',error.message));
