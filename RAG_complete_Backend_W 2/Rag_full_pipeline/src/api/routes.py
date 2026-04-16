import os, json, uuid, asyncio, logging, time, threading
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from src.config import MQ_QUEUE_PRIORITY, MQ_QUEUE_NORMAL, MQ_QUEUE_LARGE, MQ_QUEUE_DEAD, UPLOAD_DIR, cfg
from src.models.schemas import BatchSession, FileProgress, JobPayload

logger = logging.getLogger(__name__)
DEFAULT_ID = "00000000-0000-0000-0000-000000000000"

def create_router(rsm, ids, pipeline, mq_conn):
    router = APIRouter()
    
    # ▶ STORAGE ROUTES ────────────────────────────────────────────────────────
    
    @router.get("/storage/health")
    async def storage_health():
        if not pipeline.storage:
            return {"seaweedfs": "not configured"}
        return await pipeline.storage.health()

    @router.get("/storage/jobs/{job_id}/files")
    async def list_job_files(job_id: str):
        """List all SeaweedFS objects associated with a job (raw, processed, chunks)."""
        if not pipeline.storage:
            raise HTTPException(501, "Object storage not configured")
        return await pipeline.storage.list_job_files(job_id)

    @router.delete("/storage/jobs/{job_id}/files")
    async def delete_job_files(job_id: str):
        """Remove all SeaweedFS artefacts for a completed or failed job."""
        if not pipeline.storage:
            raise HTTPException(501, "Object storage not configured")
        deleted = await pipeline.storage.delete_job_artefacts(job_id)
        return {"deleted_count": deleted}

    @router.get("/storage/jobs/{job_id}/pdf-url")
    async def get_pdf_url(job_id: str, filename: str):
        """Return the direct SeaweedFS filer URL for the raw PDF of a job."""
        if not pipeline.storage:
            raise HTTPException(501, "Object storage not configured")
        return {"url": pipeline.storage.pdf_url(job_id, filename)}


    @router.post("/upload/pdfs")
    async def upload_files(files: List[UploadFile] = File(...), user_id: str = Form(""), dept_id: str = Form(""), upload_type: str = Form("user"), chat_id: str = Form("")):
        try:
            sid = str(uuid.uuid4())
            # Ensure we use a valid UID, falling back to the System account if none provided or invalid
            uid = user_id if (user_id and user_id != DEFAULT_ID) else ids.get("user_default") or DEFAULT_ID
            did = dept_id if (dept_id and dept_id != DEFAULT_ID) else ids.get("dept_default") or DEFAULT_ID
            cid = chat_id or None
            if not rsm or not rsm.ping(): raise HTTPException(503, "Redis is offline")

            max_bytes = int(cfg.max_pdf_size_mb * 1024 * 1024)
            session = BatchSession(session_id=sid, total=len(files), user_id=str(uid), dept_id=str(did), upload_type=upload_type)
            rsm.create_session(session)

            for f in files:
                contents = await f.read()
                fid = str(uuid.uuid4())

                # Validate file size
                if len(contents) > max_bytes:
                    raise HTTPException(413, f"File '{f.filename}' exceeds {cfg.max_pdf_size_mb}MB limit")

                fpath = UPLOAD_DIR / f"{fid}_{f.filename}"
                fpath.write_bytes(contents)

                # Register upload record in PostgreSQL BEFORE publishing job (FK integrity)
                upload_id = None
                try:
                    if upload_type == "admin":
                        upload_id = pipeline.rbac.register_admin_upload(
                            admin_user_id=str(uid), dept_id=str(did),
                            file_name=f.filename, file_path=str(fpath),
                            file_size_bytes=len(contents),
                        )
                    else:
                        upload_id = pipeline.rbac.register_user_upload(
                            user_id=str(uid), dept_id=str(did),
                            file_name=f.filename, file_path=str(fpath),
                            chat_id=cid, file_size_bytes=len(contents),
                            upload_scope="chat" if cid else "dept",
                        )
                except Exception as reg_err:
                    logger.error(f"Failed to register upload for {f.filename}: {reg_err}")

                fp = FileProgress(file_id=fid, session_id=sid, filename=f.filename, size_kb=len(contents)/1024)
                rsm.register_file(sid, fp)
                job = JobPayload(
                    session_id=sid, file_id=fid, filename=f.filename,
                    file_path=str(fpath), file_size_kb=len(contents)/1024,
                    user_id=str(uid), dept_id=str(did),
                    upload_type=upload_type,
                    chat_id=str(cid) if cid else None,
                    upload_id=upload_id,
                )
                from src.database.rabbitmq_broker import publish_job
                publish_job(job)
            return JSONResponse({"session_id": sid, "files": len(files)})
        except HTTPException: raise
        except Exception as e: 
            logger.error(f"Upload failed: {e}", exc_info=True); raise HTTPException(500, detail=str(e))

    @router.get("/upload/progress/{session_id}")
    async def upload_progress(session_id: str):
        try:
            if not rsm or not rsm.ping(): raise HTTPException(503, "Redis is offline")

            async def _gen():
                summary = rsm.session_summary(session_id)
                if summary:
                    for f in summary.get("files", []):
                        yield f"data: {json.dumps({'type': 'file_progress', 'data': f})}\n\n"

                # Bridge blocking Redis subscribe into async generator via queue
                q = asyncio.Queue()
                loop = asyncio.get_running_loop()

                def _blocking_subscribe():
                    try:
                        for event in rsm.subscribe_session(session_id):
                            loop.call_soon_threadsafe(q.put_nowait, event)
                    except Exception as sub_err:
                        logger.warning(f"Subscribe error: {sub_err}")
                    finally:
                        loop.call_soon_threadsafe(q.put_nowait, None)

                thread = threading.Thread(target=_blocking_subscribe, daemon=True)
                thread.start()

                while True:
                    event = await q.get()
                    if event is None:
                        break
                    yield f"data: {json.dumps(event)}\n\n"

            return StreamingResponse(_gen(), media_type="text/event-stream")
        except Exception as e:
            raise HTTPException(500, detail=str(e))

    @router.post("/query")
    async def rag_query(question: str = Form(...), user_id: str = Form(""), dept_id: str = Form(""), chat_id: str = Form(""), search: str = Form("hybrid")):
        try:
            uid = user_id or ids.get("user_default") or DEFAULT_ID
            did = dept_id or ids.get("dept_default") or DEFAULT_ID
            if not chat_id or chat_id == "null" or chat_id == "": chat_id = pipeline.rbac.create_chat(uid, did, title=question[:50])
            res = pipeline.query(question, str(uid), str(did), chat_id, search)
            res["chat_id"] = chat_id; return JSONResponse(res)
        except Exception as e: 
            logger.error(f"Query Error: {e}", exc_info=True); raise HTTPException(500, detail=str(e))

    @router.get("/admin/chats")
    async def get_all_chats(user_id: str = DEFAULT_ID, dept_id: str = DEFAULT_ID):
        conn = pipeline.rbac._get_conn()
        try:
            cur = pipeline.rbac._cur(conn)
            cur.execute("SELECT is_super_admin FROM users WHERE id=%s", (user_id,))
            user = cur.fetchone(); is_super = user["is_super_admin"] if user else False
            if is_super:
                cur.execute("SELECT c.*, u.name as user_name, d.name as dept_name FROM chat c JOIN users u ON u.id=c.user_id JOIN departments d ON d.id=c.department_id ORDER BY c.created_at DESC")
            else:
                cur.execute("SELECT c.*, u.name as user_name, d.name as dept_name FROM chat c JOIN users u ON u.id=c.user_id JOIN departments d ON d.id=c.department_id WHERE c.department_id=%s ORDER BY c.created_at DESC", (dept_id,))
            rows = cur.fetchall()
            # Convert all UUID/Timestamp to strings for JSON safety
            chats = []
            for r in rows:
                item = dict(r)
                for k, v in item.items():
                    if hasattr(v, "hex") or not isinstance(v, (str, int, float, bool, type(None))): item[k] = str(v)
                chats.append(item)
            cur.close(); return JSONResponse(chats)
        except Exception as e:
            logger.error(f"Failed to fetch chats: {e}", exc_info=True)
            raise HTTPException(500, detail=str(e))
        finally:
            pipeline.rbac._put_conn(conn)

    @router.get("/admin/audit")
    async def get_audit(dept_id: str = None): return JSONResponse(pipeline.rbac.get_audit_log(dept_id))

    @router.post("/admin/grant")
    async def grant_access(granting_dept_id: str = Form(...), receiving_dept_id: str = Form(...), user_id: str = Form(...), access_type: str = Form("read")):
        try:
            grant_id = pipeline.rbac.grant_dept_access(granting_dept_id, receiving_dept_id, user_id, access_type)
            return JSONResponse({"status": "success", "grant_id": grant_id})
        except Exception as e:
            logger.error(f"Failed to grant access: {e}", exc_info=True)
            raise HTTPException(500, detail=str(e))

    @router.get("/status")
    async def pstatus():
        conn = pipeline.rbac._get_conn()
        try:
            cur = pipeline.rbac._cur(conn)
            
            # 1. Fetch Users (Excluding System Master)
            cur.execute("SELECT u.id, u.name, u.is_super_admin, d.name as dept_name, d.id as dept_id FROM users u JOIN departments d ON d.id=u.department_id WHERE u.email != 'system@internal.rag'")
            users = []
            for r in cur.fetchall():
                item = dict(r)
                for k, v in item.items():
                    if not isinstance(v, (str, int, float, bool, type(None))):
                        item[k] = str(v)
                users.append(item)
            
            # 2. Fetch Departments (For Admin Dropdown)
            cur.execute("SELECT id, name FROM departments WHERE name != 'System' ORDER BY name ASC")
            depts = []
            for r in cur.fetchall():
                item = dict(r)
                for k, v in item.items():
                    if not isinstance(v, (str, int, float, bool, type(None))):
                        item[k] = str(v)
                depts.append(item)
            
            cur.close()
            s = {
                "redis_online": bool(rsm and rsm.ping()), 
                "rabbitmq_online": bool(mq_conn and mq_conn.is_open), 
                "users": users,
                "departments": depts
            }
            return JSONResponse(s)
        except Exception as e: return JSONResponse({"error": str(e)}, status_code=200)
        finally:
            pipeline.rbac._put_conn(conn)

    @router.get("/ui", response_class=HTMLResponse)
    async def upload_ui():
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise RAG Pipeline</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #3b82f6;
            --primary-glow: rgba(59, 130, 246, 0.5);
            --secondary: #6366f1;
            --bg-dark: #020617;
            --bg-card: #0f172a;
            --bg-overlay: rgba(15, 23, 42, 0.7);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: rgba(255, 255, 255, 0.1);
            --success: #10b981;
            --error: #ef4444;
            --warning: #f59e0b;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Outfit', 'Inter', sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            overflow-x: hidden;
            background-image: radial-gradient(circle at 50% -20%, #1e293b 0%, #020617 80%);
            min-height: 100vh;
        }

        .app-container {
            display: flex;
            height: 100vh;
            max-width: 100vw;
        }

        /* Sidebar Navigation */
        nav {
            width: 260px;
            background: var(--bg-card);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            padding: 1.5rem;
            flex-shrink: 0;
            z-index: 100;
        }

        .logo {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 2.5rem;
            background: linear-gradient(to right, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
        }

        .nav-section { margin-bottom: 2rem; }
        .nav-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            color: var(--text-muted);
            letter-spacing: 0.1em;
            margin-bottom: 1rem;
            font-weight: 600;
        }

        .nav-item {
            display: flex;
            align-items: center;
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
            border-radius: 0.75rem;
            color: var(--text-muted);
            text-decoration: none;
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            font-weight: 500;
        }

        .nav-item:hover {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-main);
        }

        .nav-item.active {
            background: rgba(59, 130, 246, 0.15);
            color: var(--primary);
            box-shadow: inset 0 0 0 1px rgba(59, 130, 246, 0.2);
        }

        /* User Profile & Switcher at Bottom */
        .user-panel {
            margin-top: auto;
            border-top: 1px solid var(--border);
            padding-top: 1.5rem;
        }

        .user-info { margin-bottom: 1rem; }
        .user-name { font-weight: 600; display: block; overflow: hidden; text-overflow: ellipsis; }
        .user-role { font-size: 0.75rem; color: var(--text-muted); }
        
        select {
            background: #1e293b;
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 0.6rem;
            border-radius: 0.5rem;
            width: 100%;
            font-size: 0.85rem;
            outline: none;
        }

        /* Main Content View */
        main {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            position: relative;
        }

        header {
            padding: 1.5rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            backdrop-filter: blur(12px);
            background: var(--bg-overlay);
            z-index: 50;
        }

        .header-title { font-size: 1.25rem; font-weight: 600; }

        .view-pane {
            flex: 1;
            padding: 2rem;
            overflow-y: auto;
            display: none;
        }

        .view-pane.active { display: block; }

        /* Dashboard/Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            padding: 1.5rem;
            border-radius: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .stat-val { font-size: 1.75rem; font-weight: 700; margin: 0.5rem 0; color: var(--primary); }
        .stat-lbl { font-size: 0.85rem; color: var(--text-muted); }

        /* Upload Section */
        .upload-zone {
            border: 2px dashed var(--border);
            border-radius: 1.5rem;
            padding: 4rem 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(255, 255, 255, 0.02);
            margin-bottom: 2rem;
        }

        .upload-zone:hover {
            border-color: var(--primary);
            background: rgba(59, 130, 246, 0.05);
        }

        .upload-title { font-size: 1.25rem; font-weight: 600; margin-bottom: 0.5rem; }
        .upload-hint { color: var(--text-muted); font-size: 0.9rem; }

        .file-list { margin-top: 2rem; }
        .progress-tile {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 1rem;
            padding: 1rem 1.5rem;
            margin-bottom: 1rem;
            animation: slideIn 0.3s ease-out;
        }

        @keyframes slideIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

        .tile-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; }
        .tile-name { font-weight: 500; font-size: 0.95rem; }
        .tile-stage { font-size: 0.7rem; text-transform: uppercase; font-weight: 700; border-radius: 4px; padding: 2px 8px; background: rgba(59, 130, 246, 0.1); color: var(--primary); }

        .progress-bar-bg { height: 6px; background: rgba(255, 255, 255, 0.1); border-radius: 3px; overflow: hidden; }
        .progress-bar-fill { height: 100%; background: var(--primary); transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1); width: 0%; box-shadow: 0 0 10px var(--primary-glow); }

        /* Chat View */
        .chat-layout { height: 100%; display: flex; flex-direction: column; overflow: hidden; max-width: 900px; margin: 0 auto; }
        #chat-scroll { flex: 1; overflow-y: auto; padding-right: 10px; margin-bottom: 1.5rem; }
        .bubble { max-width: 80%; padding: 1rem 1.25rem; border-radius: 1.25rem; margin-bottom: 1rem; line-height: 1.6; font-size: 0.95rem; }
        .bubble-bot { align-self: flex-start; background: #1e293b; border-bottom-left-radius: 4px; }
        .bubble-user { align-self: flex-end; background: var(--primary); color: white; border-bottom-right-radius: 4px; margin-left: auto; }

        .chat-input-area { position: relative; }
        textarea {
            width: 100%; resize: none; background: #1e293b; border: 1px solid var(--border); border-radius: 1rem;
            padding: 1rem 3.5rem 1rem 1.25rem; color: white; font-family: inherit; font-size: 1rem; outline: none;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }

        .send-btn {
            position: absolute; right: 0.75rem; bottom: 0.75rem; width: 2.5rem; height: 2.5rem; border-radius: 0.75rem;
            background: var(--primary); border: none; color: white; cursor: pointer; display: flex; align-items: center; justify-content: center;
        }

        /* Buttons */
        .btn-primary {
            background: var(--primary); color: white; border: none; padding: 0.8rem 1.5rem; border-radius: 0.75rem;
            font-weight: 600; cursor: pointer; transition: all 0.2s;
        }
        .btn-primary:hover { transform: translateY(-1px); filter: brightness(1.1); }

        /* Typing Dots */
        .typing { display: flex; gap: 4px; padding: 1rem; background: #1e293b; border-radius: 1.25rem; width: fit-content; margin-bottom: 1rem; }
        .dot { width: 6px; height: 6px; background: var(--text-muted); border-radius: 50%; animation: bounce 1.4s infinite ease-in-out both; }
        .dot:nth-child(1) { animation-delay: -0.32s; } .dot:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
    </style>
</head>
<body>
    <div class="app-container">
        <nav>
            <div class="logo">RAG.Engine</div>
            
            <div class="nav-section">
                <div class="nav-label">General</div>
                <div class="nav-item active" onclick="switchPane('pane-chat', this)">
                    <span>Chat Assistant</span>
                </div>
                <div class="nav-item" onclick="switchPane('pane-upload', this)">
                    <span>Data Ingestion</span>
                </div>
            </div>

            <div class="nav-section" id="nav-admin" style="display:none">
                <div class="nav-label">Administration</div>
                <div class="nav-item" onclick="switchPane('pane-audit', this)">
                    <span>Audit Logs</span>
                </div>
                <div class="nav-item" onclick="switchPane('pane-access', this)">
                    <span>Policy Access</span>
                </div>
            </div>

            <div class="user-panel">
                <div class="user-info">
                    <span class="user-name" id="label-user">Loading...</span>
                    <span class="user-role" id="label-role">Account Placeholder</span>
                </div>
                <select id="user-switcher"></select>
            </div>
        </nav>

        <main>
            <header>
                <div class="header-title" id="pane-title">Chat Assistant</div>
                <div id="status-indicators" style="display:flex; gap: 1rem;">
                    <div id="status-redis" style="font-size:0.7rem; color: var(--success);">● Redis Online</div>
                    <div id="status-mq" style="font-size:0.7rem; color: var(--success);">● MQ Connected</div>
                </div>
            </header>

            <!-- Chat Pane -->
            <div id="pane-chat" class="view-pane active">
                <div class="chat-layout">
                    <div id="chat-scroll">
                        <div class="bubble bubble-bot">Welcome. I'm your enterprise knowledge agent. Upload documents to get started or ask me questions about your processed records.</div>
                    </div>
                    <div id="typing-box" style="display:none">
                        <div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
                    </div>
                    <div class="chat-input-area">
                        <textarea id="chat-input" rows="1" placeholder="Type your message..."></textarea>
                        <button class="send-btn" id="send-btn">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                        </button>
                    </div>
                </div>
            </div>

            <!-- Upload Pane -->
            <div id="pane-upload" class="view-pane">
                <div class="stats-grid">
                    <div class="stat-card"><div class="stat-lbl">Active Workers</div><div class="stat-val" id="stat-workers">0</div></div>
                    <div class="stat-card"><div class="stat-lbl">Successful Ingests</div><div class="stat-val" id="stat-ok">0</div></div>
                    <div class="stat-card"><div class="stat-lbl">Failed Jobs</div><div class="stat-val" id="stat-fail">0</div></div>
                </div>

                <!-- ONLY VISIBLE TO SUPER ADMIN -->
                <div id="admin-dept-scope" class="stat-card" style="margin-bottom: 2rem; display: none;">
                    <h3 style="margin-bottom: 1rem;">Department Scope (Admin Override)</h3>
                    <select id="target-dept"></select>
                    <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.5rem;">As an administrator, you can route these documents to any department's index.</p>
                </div>

                <input type="file" id="file-input" multiple hidden accept="application/pdf">
                <div class="upload-zone" onclick="document.getElementById('file-input').click()">
                    <div class="upload-title">Drop files or click to browse</div>
                    <div class="upload-hint">Upload high-resolution PDFs for optimized OCR + Embedding</div>
                </div>

                <div class="file-list" id="progress-list"></div>
            </div>

            <!-- Audit Pane -->
            <div id="pane-audit" class="view-pane">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 2rem;">
                    <h3>System Audit Logs</h3>
                    <button class="btn-primary" onclick="loadAudit()" style="padding: 0.4rem 1rem; font-size: 0.8rem;">Refresh</button>
                </div>
                <div id="audit-table-container"></div>
            </div>

            <!-- Access Pane -->
            <div id="pane-access" class="view-pane">
                <h3>Department Security Policy</h3>
                <div class="stat-card" style="margin-top: 1.5rem; max-width: 600px;">
                    <p style="margin-bottom: 1.5rem; color: var(--text-muted);">Grant cross-department access to allow users from one department to query knowledge owned by another.</p>
                    <div style="margin-bottom:1rem">
                        <label style="font-size:0.75rem; display:block; margin-bottom:5px">DATA OWNER (Granting Dept)</label>
                        <select id="grant-from"></select>
                    </div>
                    <div style="margin-bottom:1.5rem">
                        <label style="font-size:0.75rem; display:block; margin-bottom:5px">RECEIVING PARTY (Target Dept)</label>
                        <select id="grant-to"></select>
                    </div>
                    <button class="btn-primary" id="grant-btn">Apply Policy Change</button>
                </div>
            </div>

        </main>
    </div>

    <script>
        const $ = id => document.getElementById(id);
        let currentUser = null;
        let allUsers = [];
        let allDepts = [];
        let isTyping = false;
        let currentChatId = null;

        // Pane Switcher
        function switchPane(paneId, navItem) {
            document.querySelectorAll('.view-pane').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
            $(paneId).classList.add('active');
            if (navItem) navItem.classList.add('active');
            
            const titles = {
                'pane-chat': 'Chat Assistant',
                'pane-upload': 'Data Ingestion',
                'pane-audit': 'System Audit Logs',
                'pane-access': 'Policy Management'
            };
            $('pane-title').innerText = titles[paneId] || 'Dashboard';
            
            if(paneId === 'pane-audit') loadAudit();
        }

        async function loadStatus() {
            try {
                const r = await fetch('/status');
                const d = await r.json();
                allUsers = d.users || [];
                allDepts = d.departments || [];
                
                $('status-redis').style.color = d.redis_online ? 'var(--success)' : 'var(--error)';
                $('status-mq').style.color = d.rabbitmq_online ? 'var(--success)' : 'var(--error)';
                
                const sel = $('user-switcher');
                sel.innerHTML = allUsers.map((u, i) => `<option value="${i}">${u.name}</option>`).join('');
                
                const deptSelectors = [$('target-dept'), $('grant-from'), $('grant-to')];
                deptSelectors.forEach(s => {
                    if(s) s.innerHTML = allDepts.map(dt => `<option value="${dt.id}">${dt.name}</option>`).join('');
                });

                sel.onchange = () => applyUser(allUsers[sel.value]);
                if(allUsers.length) applyUser(allUsers[0]);
            } catch(e) { console.error(e); }
        }

        function applyUser(user) {
            currentUser = user;
            $('label-user').innerText = user.name;
            $('label-role').innerText = user.is_super_admin ? 'Super Administrator' : (user.dept_name || 'Dept Member');
            $('nav-admin').style.display = user.is_super_admin ? 'block' : 'none';
            
            // Toggle Admin Override visibility - ONLY for Super Admin
            if(user.is_super_admin) {
                $('admin-dept-scope').style.display = 'block';
            } else {
                $('admin-dept-scope').style.display = 'none';
            }

            // Clear view
            $('chat-scroll').innerHTML = '<div class="bubble bubble-bot">Context cleared. Session switched to ' + user.name + '.</div>';
            currentChatId = null;
        }

        // --- UPLOAD LOGIC ---
        $('file-input').onchange = async (e) => {
            const files = e.target.files;
            if(!files.length) return;
            
            // Determine target dept id based on role
            const targetDeptId = currentUser.is_super_admin ? $('target-dept').value : currentUser.dept_id;

            const fd = new FormData();
            for(const f of files) fd.append('files', f);
            fd.append('user_id', currentUser.id);
            fd.append('dept_id', targetDeptId); 
            fd.append('upload_type', currentUser.is_super_admin ? 'admin' : 'user');

            try {
                const r = await fetch('/upload/pdfs', { method:'POST', body:fd });
                const d = await r.json();
                startProgress(d.session_id);
            } catch(e) { alert('Upload error'); }
        };

        function startProgress(sid) {
            const ev = new EventSource('/upload/progress/' + sid);
            ev.onmessage = e => {
                const msg = JSON.parse(e.data);
                if(msg.type === 'file_progress') {
                    updateTile(msg.data);
                } else if(msg.type === 'session_complete') {
                    ev.close();
                    refreshStats();
                }
            };
        }

        function updateTile(f) {
            let tile = $(f.file_id);
            if(!tile) {
                tile = document.createElement('div');
                tile.id = f.file_id;
                tile.className = 'progress-tile';
                $('progress-list').prepend(tile);
            }
            tile.innerHTML = `
                <div class="tile-header">
                    <span class="tile-name">${f.filename}</span>
                    <span class="tile-stage">${f.stage}</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: ${f.pct}%"></div>
                </div>
            `;
        }

        async function refreshStats() {
            try {
                const d = await (await fetch('/status')).json();
                $('stat-ok').innerText = d.redis_online ? 'Online' : 'Checking...';
            } catch(e) { console.error(e); }
        }

        // --- CHAT LOGIC ---
        $('send-btn').onclick = async () => {
            const msg = $('chat-input').value.trim();
            if(!msg || isTyping) return;
            
            addBubble('user', msg);
            $('chat-input').value = '';
            $('typing-box').style.display = 'block';
            isTyping = true; scrollChat();

            const fd = new FormData();
            fd.append('question', msg);
            fd.append('user_id', currentUser.id);
            fd.append('dept_id', currentUser.dept_id);
            if(currentChatId) fd.append('chat_id', currentChatId);

            try {
                const r = await fetch('/query', { method: 'POST', body: fd });
                const d = await r.json();
                if(d.chat_id) currentChatId = d.chat_id;
                addBubble('bot', d.answer, d.citations);
            } catch(e) {
                addBubble('bot', 'Execution error encountered.');
            } finally {
                $('typing-box').style.display = 'none';
                isTyping = false; scrollChat();
            }
        };

        function addBubble(role, text, sources=[]) {
            const b = document.createElement('div');
            b.className = 'bubble bubble-' + role;
            b.innerText = text;
            if(sources && sources.length > 0) {
                const s = document.createElement('div');
                s.style.cssText = 'font-size:0.7rem; margin-top:0.5rem; color: #60a5fa; border-top: 1px solid rgba(255,255,255,0.05); padding-top:0.5rem;';
                s.innerText = 'Sources: ' + sources.join(', ');
                b.appendChild(s);
            }
            $('chat-scroll').appendChild(b);
        }

        function scrollChat() { $('chat-scroll').scrollTop = $('chat-scroll').scrollHeight; }

        async function loadAudit() {
            const r = await fetch('/admin/audit');
            const logs = await r.json();
            $('audit-table-container').innerHTML = `
                <table style="width:100%; border-collapse:collapse; font-size:0.85rem;">
                    <thead style="background:rgba(255,255,255,0.05); text-align:left;">
                        <tr><th style="padding:1rem;">Time</th><th style="padding:1rem;">Action</th><th style="padding:1rem;">Target</th></tr>
                    </thead>
                    <tbody>
                        ${logs.map(l => `
                            <tr style="border-bottom: 1px solid var(--border);">
                                <td style="padding:1rem; color:var(--text-muted);">${l.created_at.split('.')[0]}</td>
                                <td style="padding:1rem; font-weight:600;">${l.action_type}</td>
                                <td style="padding:1rem; color:var(--primary);">${l.target_type || '-'} (${l.target_id || 'n/a'})</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        }

        $('grant-btn').onclick = async () => {
            const fd = new FormData();
            fd.append('granting_dept_id', $('grant-from').value);
            fd.append('receiving_dept_id', $('grant-to').value);
            fd.append('user_id', currentUser.id);
            fd.append('access_type', 'read');
            
            const r = await fetch('/admin/grant', { method: 'POST', body: fd });
            if(r.ok) alert('Cross-dept access policy updated.');
            else alert('Policy update failed.');
        }

        loadStatus();
    </script>
</body>
</html>
"""
        return HTMLResponse(content=html)

    return router
