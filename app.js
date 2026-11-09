import { FilesetResolver, PoseLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const video = document.querySelector("#video");
const canvas = document.querySelector("#overlay");
const imagePreview = document.querySelector("#imagePreview");
const ctx = canvas.getContext("2d");
const startBtn = document.querySelector("#startBtn");
const stopBtn = document.querySelector("#stopBtn");
const imageInput = document.querySelector("#imageInput");
const statusEl = document.querySelector("#status");
const messageEl = document.querySelector("#message");
const placeholder = document.querySelector("#placeholder");
const countEl = document.querySelector("#landmarkCount");
const fpsEl = document.querySelector("#fps");
const anglesEl = document.querySelector("#angles");

let landmarker = null;
let stream = null;
let running = false;
let lastVideoTime = -1;
let lastFrameAt = 0;

const CONNECTIONS = [
  [11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],
  [12,14],[14,16],[16,18],[16,20],[16,22],[18,20],
  [11,23],[12,24],[23,24],[23,25],[25,27],[27,29],[29,31],[27,31],
  [24,26],[26,28],[28,30],[30,32],[28,32]
];
const JOINTS = {
  "Left elbow":[11,13,15], "Right elbow":[12,14,16],
  "Left shoulder":[13,11,23], "Right shoulder":[14,12,24],
  "Left hip":[11,23,25], "Right hip":[12,24,26],
  "Left knee":[23,25,27], "Right knee":[24,26,28]
};

function angle(a,b,c) {
  const v1=[a.x-b.x,a.y-b.y], v2=[c.x-b.x,c.y-b.y];
  const d=Math.hypot(...v1)*Math.hypot(...v2);
  if (!d) return null;
  const cosine=Math.max(-1,Math.min(1,(v1[0]*v2[0]+v1[1]*v2[1])/d));
  return Math.round(Math.acos(cosine)*180/Math.PI);
}
function setAngles(landmarks) {
  anglesEl.innerHTML="";
  for (const [name,ids] of Object.entries(JOINTS)) {
    const value=landmarks ? angle(...ids.map(i=>landmarks[i])) : null;
    const card=document.createElement("div");
    card.className="angle";
    const label=document.createElement("span");
    label.textContent=name;
    const number=document.createElement("strong");
    number.textContent=value===null ? "—" : `${value}°`;
    card.append(label,number);
    anglesEl.append(card);
  }
}
function resizeCanvas(width,height) {
  if (canvas.width!==width || canvas.height!==height) {
    canvas.width=width; canvas.height=height;
  }
}
function drawPose(landmarks, width, height) {
  resizeCanvas(width,height);
  ctx.clearRect(0,0,width,height);
  if (!landmarks) { countEl.textContent="0 / 33"; setAngles(null); return; }
  const visible=landmarks.filter(p=>(p.visibility ?? 1)>.35);
  countEl.textContent=`${visible.length} / 33`;
  ctx.lineWidth=Math.max(2,width/320);
  ctx.strokeStyle="#b7f34b";
  for (const [a,b] of CONNECTIONS) {
    if ((landmarks[a].visibility ?? 1)<.35 || (landmarks[b].visibility ?? 1)<.35) continue;
    ctx.beginPath();
    ctx.moveTo(landmarks[a].x*width,landmarks[a].y*height);
    ctx.lineTo(landmarks[b].x*width,landmarks[b].y*height);
    ctx.stroke();
  }
  ctx.fillStyle="#55d6ff";
  for (const p of visible) {
    ctx.beginPath(); ctx.arc(p.x*width,p.y*height,Math.max(3,width/180),0,Math.PI*2); ctx.fill();
  }
  setAngles(landmarks);
}
async function loadModel() {
  if (landmarker) return landmarker;
  statusEl.textContent="Loading model…";
  messageEl.textContent="Loading the pose detection model. This may take a moment.";
  const vision=await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  landmarker=await PoseLandmarker.createFromOptions(vision,{
    baseOptions:{
      modelAssetPath:"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate:"GPU"
    },
    runningMode:"VIDEO",
    numPoses:1
  });
  return landmarker;
}
async function startCamera() {
  try {
    await loadModel();
    stream=await navigator.mediaDevices.getUserMedia({
      audio:false, video:{facingMode:"user",width:{ideal:1280},height:{ideal:720}}
    });
    imagePreview.hidden=true;
    video.style.display="block";
    video.srcObject=stream;
    await video.play();
    placeholder.hidden=true;
    running=true; lastVideoTime=-1;
    startBtn.disabled=true; stopBtn.disabled=false;
    statusEl.textContent="Camera active";
    messageEl.textContent="Pose tracking is running. Keep your body visible in the frame.";
    requestAnimationFrame(renderFrame);
  } catch (error) {
    statusEl.textContent="Unable to start";
    messageEl.textContent=`Could not start camera/model: ${error.message}. Check camera permission and internet connection.`;
    stopCamera();
  }
}
function renderFrame(now) {
  if (!running) return;
  if (video.readyState>=2 && video.currentTime!==lastVideoTime) {
    lastVideoTime=video.currentTime;
    try {
      const result=landmarker.detectForVideo(video,now);
      const pose=result.landmarks?.[0] ?? null;
      drawPose(pose,video.videoWidth,video.videoHeight);
      if (lastFrameAt) fpsEl.textContent=`${Math.round(1000/(now-lastFrameAt))} FPS`;
      lastFrameAt=now;
    } catch (error) {
      messageEl.textContent=`Detection error: ${error.message}`;
    }
  }
  requestAnimationFrame(renderFrame);
}
function stopCamera() {
  running=false;
  if (stream) stream.getTracks().forEach(track=>track.stop());
  stream=null; video.srcObject=null;
  video.style.display="block"; imagePreview.hidden=true; imagePreview.removeAttribute("src");
  ctx.clearRect(0,0,canvas.width,canvas.height);
  placeholder.hidden=false;
  startBtn.disabled=false; stopBtn.disabled=true;
  if (statusEl.textContent==="Camera active") statusEl.textContent="Stopped";
  countEl.textContent="0 / 33"; fpsEl.textContent="—"; setAngles(null);
}
async function analyzeImage(file) {
  if (!file) return;
  try {
    await loadModel();
    const bitmap=await createImageBitmap(file);
    landmarker.setOptions({runningMode:"IMAGE"});
    const result=landmarker.detect(bitmap);
    resizeCanvas(bitmap.width,bitmap.height);
    placeholder.hidden=true;
    video.pause();
    video.srcObject=null;
    video.style.display="none";
    imagePreview.src=URL.createObjectURL(file);
    imagePreview.hidden=false;
    const pose=result.landmarks?.[0] ?? null;
    drawPose(pose,bitmap.width,bitmap.height);
    statusEl.textContent=pose ? "Image analyzed" : "No pose found";
    messageEl.textContent=pose ? "Image processed locally in your browser." : "No person was detected. Try a clearer, full-body image.";
    bitmap.close();
    landmarker.setOptions({runningMode:"VIDEO"});
  } catch (error) {
    messageEl.textContent=`Image analysis failed: ${error.message}`;
    statusEl.textContent="Analysis failed";
  } finally {
    imageInput.value="";
  }
}
startBtn.addEventListener("click",startCamera);
stopBtn.addEventListener("click",stopCamera);
imageInput.addEventListener("change",event=>analyzeImage(event.target.files?.[0]));
window.addEventListener("pagehide",stopCamera);
setAngles(null);
