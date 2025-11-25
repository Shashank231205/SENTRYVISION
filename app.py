import gradio as gr, os, json, cv2, tempfile, torch
from pathlib import Path
from utils.video_loader import load_video
from utils.mask_overlay import overlay_masks
from utils.bounding_boxes import draw_boxes
from utils.timeline import generate_timeline, save_json
from utils.compare_videos import compare
from utils.audio_transcription import extract_audio, transcribe
from engine.segmentation import extract_video_masks
from engine.detection import extract_video_boxes
from engine.flow import track_motion
from engine.pose import load_pose_model, run_pose_estimation
from engine.highlights import extract_highlights
from engine.chapters import detect_chapters, save_chapters_json
from engine.keyframes import detect_shot_changes, save_keyframes
from engine.camera_motion import analyze_camera_motion
from engine.realtime.stream_processor import process_live_frame
from engine.realtime.webcam_reader import webcam_stream
from engine.realtime.rtsp_reader import rtsp_stream
from engine.pose_3d.lift2d_to_3d import lift_2d_to_3d
from engine.pose_3d.metrics_3d import compute_joint_angles_3d
from engine.depth.depth_map import estimate_depth_from_spatial
from engine.depth.pointcloud import depth_to_pointcloud
from engine.depth.mesh_reconstruct import create_mesh_from_pointcloud
from engine.actions.classifier import classify_actions
from engine.actions.timeline_actions import actions_to_timeline
from engine.gaze.face_landmarks import detect_face_landmarks
from engine.gaze.gaze_estimator import estimate_gaze
from engine.gaze.eye_contact import detect_eye_contact
from engine.vqa.qa import answer_question
from engine.captioning.caption import generate_video_caption
from engine.summary.summarizer import summarize_video
from models.pe_core import load_pe_core
from models.pe_spatial import load_pe_spatial
from models.plm import load_plm
from personas import PERSONA_PROMPTS
from context_prompts import CONTEXT_PROMPT

pe_core=load_pe_core()
pe_spatial=load_pe_spatial()
plm=load_plm()
pose_model=load_pose_model()

def get_frame_sizes(v):
    cap=cv2.VideoCapture(v);a=[]
    while True:
        r,f=cap.read()
        if not r:break
        a.append((f.shape[0],f.shape[1]))
    cap.release();return a

def get_background():
    p=Path("style/background.html")
    return p.read_text() if p.exists() else ""

def render_overlay_video(v,m,b,p):
    t=tempfile.mkdtemp();o=os.path.join(t,"overlay.mp4")
    c=cv2.VideoCapture(v);fps=c.get(5);w,h=int(c.get(3)),int(c.get(4))
    wri=cv2.VideoWriter(o,cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h));i=0
    while True:
        r,f=c.read()
        if not r:break
        if i<len(m):f=overlay_masks(f,m[i])
        if i<len(b):f=draw_boxes(f,b[i])
        if i<len(p) and p[i] is not None:
            from engine.pose import draw_skeleton
            f=draw_skeleton(f,p[i])
        wri.write(f);i+=1
    c.release();wri.release();return o

def analyze_video(v):
    t=load_video(v)
    with torch.no_grad(),torch.autocast("cuda"):g=pe_core.encode_video(t)
    with torch.no_grad(),torch.autocast("cuda"):s=pe_spatial.encode_video(t)
    pr="You are an advanced video-understanding AI. Provide: Summary, Objects, Actions, Ordered events, Observations."
    with torch.no_grad():r=plm.generate(video=t,prompt=pr,max_new_tokens=480,temperature=0.2)
    return{"summary":r,"global":g,"spatial":s,"tensor":t}

def ui_analyze(v):
    if v is None:return"Upload video.",None,None,None,None,None,None
    r=analyze_video(v);s=r["summary"];fs=get_frame_sizes(v)
    m=extract_video_masks(r["spatial"],fs);b=extract_video_boxes(m)
    tr=track_motion(v,m);tmp=tempfile.mkdtemp()
    tf=os.path.join(tmp,"traj.json");save_json(tr,tf)
    p,a=run_pose_estimation(v,pose_model)
    pf,af=os.path.join(tmp,"poses.json"),os.path.join(tmp,"angles.json")
    save_json(p,pf);save_json(a,af)
    ov=render_overlay_video(v,m,b,p)
    ev=s.split(".")[:8];tl=generate_timeline(ev)
    tlf=os.path.join(tmp,"timeline.json");save_json(tl,tlf)
    return s,ov,json.dumps(tl,indent=4),tlf,tf,pf,af

def ui_compare(a,bv):
    if not a or not bv:return"Upload two videos."
    t1,t2=load_video(a),load_video(bv)
    with torch.no_grad(),torch.autocast("cuda"):f1,f2=pe_core.encode_video(t1),pe_core.encode_video(t2)
    return f"Similarity: {float(compare(f1,f2)):.4f}"

def ui_transcribe(v):
    if not v:return"Upload video."
    return transcribe(extract_audio(v))

def ui_coach(v,p):
    if v is None:return"Upload first."
    r=analyze_video(v)
    pd=open("poses.json").read() if os.path.exists("poses.json") else"[]"
    ad=open("angles.json").read() if os.path.exists("angles.json") else"{}"
    td=open("trajectories.json").read() if os.path.exists("trajectories.json") else"[]"
    pr=PERSONA_PROMPTS[p]+f"\nPOSE:{pd}\nANGLES:{ad}\nTRAJ:{td}"
    with torch.no_grad():o=plm.generate(video=r["tensor"],prompt=pr,max_new_tokens=650,temperature=0.25)
    return o

def ui_context(v):
    if v is None:return"Upload first."
    r=analyze_video(v)
    pd=open("poses.json").read() if os.path.exists("poses.json") else"[]"
    ad=open("angles.json").read() if os.path.exists("angles.json") else"{}"
    td=open("trajectories.json").read() if os.path.exists("trajectories.json") else"[]"
    pr=f"{CONTEXT_PROMPT}\nPOSE:{pd}\nANGLES:{ad}\nTRAJ:{td}"
    with torch.no_grad():o=plm.generate(video=r["tensor"],prompt=pr,max_new_tokens=800,temperature=0.25)
    return o

def ui_highlights(v):
    if v is None:return[], "Upload first."
    if not os.path.exists("trajectories.json"):return[], "Run analysis first."
    tr=json.load(open("trajectories.json"))
    return extract_highlights(v,tr),"Done."

def ui_chapters(v):
    if v is None:return None,"Upload first."
    if not os.path.exists("trajectories.json"):return None,"Run analysis first."
    tr=json.load(open("trajectories.json"))
    ch=detect_chapters(tr);p=save_chapters_json(ch)
    return p,json.dumps(ch,indent=4)

def ui_keyframes(v):
    if v is None:return[]
    fr=detect_shot_changes(v);return save_keyframes(fr)

def ui_camera_motion(v):
    if v is None:return{}
    return analyze_camera_motion(v)

def ui_stream(m,u):
    import time
    pr=None
    src=webcam_stream() if m=="Webcam" else rtsp_stream(u)
    for f in src:
        f,pr=process_live_frame(f,pe_spatial,pose_model,pr)
        yield f;time.sleep(0.03)

def ui_pose3d(v):
    if v is None:return{}
    p,_=run_pose_estimation(v,pose_model)
    p3d=lift_2d_to_3d(p)
    ang=compute_joint_angles_3d(p3d[0] if p3d else None)
    return {"pose3d":p3d,"angles3d":ang}

def ui_depth(v):
    if v is None:return{}
    r=analyze_video(v)
    d=estimate_depth_from_spatial(r["spatial"],get_frame_sizes(v))
    pc=depth_to_pointcloud(d)
    mesh=create_mesh_from_pointcloud(pc)
    return {"depth":d,"pointcloud":pc,"mesh":mesh}

def ui_actions(v):
    if v is None:return{}
    p,_=run_pose_estimation(v,pose_model)
    ac=classify_actions(p);tl=actions_to_timeline(ac)
    return {"actions":ac,"timeline":tl}

def ui_gaze(v):
    if v is None:return{}
    lm=detect_face_landmarks(v)
    gz=estimate_gaze(lm)
    ec=detect_eye_contact(gz)
    return {"landmarks":lm,"gaze":gz,"eye_contact":ec}

def ui_vqa(v,q):
    if v is None or not q:return"Upload video + ask."
    t=load_video(v)
    return answer_question(t,q,plm)

def ui_caption(v):
    if v is None:return"Upload video."
    t=load_video(v)
    return generate_video_caption(t,plm)

def ui_summary(v,m):
    if v is None:return"Upload video."
    t=load_video(v)
    return summarize_video(t,plm,m)

with gr.Blocks(css="style/theme.css") as demo:
    gr.HTML(get_background())
    gr.Markdown("PE-VISTA 3.0 Ultra")
    with gr.Tabs():
        with gr.Tab("Analyze"):
            i=gr.Video();o1=gr.Textbox();o2=gr.Video()
            o3=gr.Code();o4=gr.File();o5=gr.File();o6=gr.File();o7=gr.File()
            gr.Button("Run").click(ui_analyze,i,[o1,o2,o3,o4,o5,o6,o7])
        with gr.Tab("Real-Time"):
            m=gr.Dropdown(["Webcam","RTSP"],value="Webcam")
            u=gr.Textbox(visible=False)
            c=gr.Image();s=gr.Button("Start")
            m.change(lambda x:gr.update(visible=x=="RTSP"),m,u)
            s.click(ui_stream,[m,u],c)
        with gr.Tab("Keyframes"):
            v=gr.Video();o=gr.Gallery()
            gr.Button("Extract").click(ui_keyframes,v,o)
        with gr.Tab("Camera Motion"):
            v=gr.Video();o=gr.JSON()
            gr.Button("Analyze").click(ui_camera_motion,v,o)
        with gr.Tab("Coach"):
            v=gr.Video();p=gr.Dropdown(choices=list(PERSONA_PROMPTS.keys()),value="cricket_batting")
            o=gr.Textbox(lines=15)
            gr.Button("Run").click(ui_coach,[v,p],o)
        with gr.Tab("Context"):
            v=gr.Video();o=gr.Textbox(lines=15)
            gr.Button("Run").click(ui_context,v,o)
        with gr.Tab("Highlights"):
            v=gr.Video();g=gr.Gallery();t=gr.Textbox()
            gr.Button("Generate").click(ui_highlights,v,[g,t])
        with gr.Tab("Chapters"):
            v=gr.Video();f=gr.File();j=gr.Code()
            gr.Button("Generate").click(ui_chapters,v,[f,j])
        with gr.Tab("3D Pose"):
            v=gr.Video();o=gr.JSON()
            gr.Button("Compute").click(ui_pose3d,v,o)
        with gr.Tab("Depth"):
            v=gr.Video();o=gr.JSON()
            gr.Button("Generate").click(ui_depth,v,o)
        with gr.Tab("Actions"):
            v=gr.Video();o=gr.JSON()
            gr.Button("Classify").click(ui_actions,v,o)
        with gr.Tab("Gaze"):
            v=gr.Video();o=gr.JSON()
            gr.Button("Analyze").click(ui_gaze,v,o)
        with gr.Tab("VQA"):
            v=gr.Video();q=gr.Textbox();o=gr.Textbox(lines=10)
            gr.Button("Ask").click(ui_vqa,[v,q],o)
        with gr.Tab("Caption"):
            v=gr.Video();o=gr.Textbox()
            gr.Button("Generate").click(ui_caption,v,o)
        with gr.Tab("Summary"):
            v=gr.Video();m=gr.Dropdown(["short","long"],value="short")
            o=gr.Textbox(lines=10)
            gr.Button("Summarize").click(ui_summary,[v,m],o)
        with gr.Tab("Compare"):
            a=gr.Video();b=gr.Video();o=gr.Textbox()
            gr.Button("Go").click(ui_compare,[a,b],o)
        with gr.Tab("Transcribe"):
            v=gr.Video();o=gr.Textbox()
            gr.Button("Run").click(ui_transcribe,v,o)
        with gr.Tab("API"):
            gr.Markdown("POST /analyze\nPOST /compare\nPOST /timeline\nPOST /trajectories\nPOST /pose\nPOST /angles\nPOST /coach\nPOST /context\nPOST /highlights\nPOST /chapters\nPOST /keyframes\nPOST /camera_motion\nPOST /pose3d\nPOST /depth\nPOST /actions\nPOST /gaze\nPOST /vqa\nPOST /caption\nPOST /summary")

demo.queue().launch(server_name="0.0.0.0",server_port=7860)
