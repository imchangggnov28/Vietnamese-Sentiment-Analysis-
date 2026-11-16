"""SQLite storage for accounts, roles and sentiment prediction history."""
import os, sqlite3, uuid, hashlib, hmac
from contextlib import closing
from datetime import datetime, timedelta, timezone
import pandas as pd
DB_PATH=os.environ.get("SENTIMENT_DB_PATH","sentiment_history.db")
VN_TZ=timezone(timedelta(hours=7))
LABEL_POS="Tích cực"; LABEL_NEG="Tiêu cực"
ROLES={"admin":"Quản trị viên","user":"Người dùng"}
SCHEMA="""
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL UNIQUE COLLATE NOCASE, password_hash TEXT NOT NULL, full_name TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'user', is_active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, source TEXT NOT NULL, batch_id TEXT, file_name TEXT, text TEXT NOT NULL, label TEXT NOT NULL, positive_prob REAL NOT NULL, negative_prob REAL NOT NULL, user_id INTEGER REFERENCES users(id));
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_label ON predictions(label);
CREATE INDEX IF NOT EXISTS idx_predictions_batch ON predictions(batch_id);
"""
def _connect():
 c=sqlite3.connect(DB_PATH,timeout=10); c.row_factory=sqlite3.Row; c.execute("PRAGMA foreign_keys=ON"); return c
def _now(): return datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S")
def _hash_password(password,salt=None):
 salt=salt or os.urandom(16); return salt.hex()+"$"+hashlib.pbkdf2_hmac("sha256",str(password).encode(),salt,260000).hex()
def _verify_password(password,stored):
 try:
  salt,digest=stored.split("$",1); return hmac.compare_digest(_hash_password(password,bytes.fromhex(salt)).split("$",1)[1],digest)
 except (ValueError,TypeError): return False
def init_db():
 with closing(_connect()) as c:
  with c:
   c.executescript(SCHEMA)
   cols={r["name"] for r in c.execute("PRAGMA table_info(predictions)")}
   if "user_id" not in cols: c.execute("ALTER TABLE predictions ADD COLUMN user_id INTEGER REFERENCES users(id)")
   c.execute("CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id)")
   if not c.execute("SELECT id FROM users WHERE username='admin'").fetchone():
    c.execute("INSERT INTO users(username,password_hash,full_name,role,is_active,created_at) VALUES(?,?,?,?,1,?)",("admin",_hash_password(os.environ.get("SENTIMENT_ADMIN_PASSWORD","Admin@123")),"Quản trị hệ thống","admin",_now()))
def authenticate(username,password):
 with closing(_connect()) as c: r=c.execute("SELECT * FROM users WHERE username=? COLLATE NOCASE",(str(username).strip(),)).fetchone()
 if r and r["is_active"] and _verify_password(password,r["password_hash"]): return {"id":r["id"],"username":r["username"],"full_name":r["full_name"],"role":r["role"]}
 return None
def create_user(username,password,full_name,role="user"):
 username=str(username).strip()
 if len(username)<3: raise ValueError("Tên đăng nhập phải có ít nhất 3 ký tự.")
 if len(str(password))<8: raise ValueError("Mật khẩu phải có ít nhất 8 ký tự.")
 if role not in ROLES: raise ValueError("Vai trò không hợp lệ.")
 with closing(_connect()) as c:
  with c: cur=c.execute("INSERT INTO users(username,password_hash,full_name,role,is_active,created_at) VALUES(?,?,?,?,1,?)",(username,_hash_password(password),str(full_name).strip() or username,role,_now())); return cur.lastrowid
def list_users():
 with closing(_connect()) as c: return pd.read_sql_query("SELECT id,username,full_name,role,is_active,created_at FROM users ORDER BY id",c)
def update_user(user_id,full_name=None,role=None,is_active=None,password=None):
 fields=[]; vals=[]
 if full_name is not None: fields.append("full_name=?"); vals.append(str(full_name).strip())
 if role is not None:
  if role not in ROLES: raise ValueError("Vai trò không hợp lệ.")
  fields.append("role=?"); vals.append(role)
 if is_active is not None: fields.append("is_active=?"); vals.append(int(bool(is_active)))
 if password:
  if len(password)<8: raise ValueError("Mật khẩu phải có ít nhất 8 ký tự.")
  fields.append("password_hash=?"); vals.append(_hash_password(password))
 if fields:
  with closing(_connect()) as c:
   with c: c.execute(f"UPDATE users SET {','.join(fields)} WHERE id=?",vals+[int(user_id)])
def save_prediction(text,label,positive_prob,negative_prob,user_id):
 with closing(_connect()) as c:
  with c: cur=c.execute("INSERT INTO predictions(created_at,source,text,label,positive_prob,negative_prob,user_id) VALUES(?,'single',?,?,?,?,?)",(_now(),str(text),label,float(positive_prob),float(negative_prob),int(user_id))); return cur.lastrowid
def save_batch(file_name,texts,labels,positive_probs,negative_probs,user_id):
 batch_id=uuid.uuid4().hex[:12]; now=_now(); rows=[(now,batch_id,file_name,str(t),lb,float(a),float(b),int(user_id)) for t,lb,a,b in zip(texts,labels,positive_probs,negative_probs)]
 with closing(_connect()) as c:
  with c: c.executemany("INSERT INTO predictions(created_at,source,batch_id,file_name,text,label,positive_prob,negative_prob,user_id) VALUES(?,'batch',?,?,?,?,?,?,?)",rows)
 return batch_id
def get_history(label=None,source=None,keyword=None,limit=100,user_id=None):
 w=[]; a=[]
 if user_id is not None: w.append("p.user_id=?"); a.append(int(user_id))
 if label: w.append("p.label=?"); a.append(label)
 if source: w.append("p.source=?"); a.append(source)
 if keyword and keyword.strip():
    like = f"%{keyword.strip()}%"
    w.append("(p.text LIKE ? OR p.file_name LIKE ? OR p.source LIKE ?)")
    a.extend([like, like, like])
 where=("WHERE "+" AND ".join(w)) if w else ""
 q=f"SELECT p.id,p.created_at,p.source,p.file_name,p.text,p.label,p.user_id,u.username,u.full_name,p.positive_prob,p.negative_prob,MAX(p.positive_prob,p.negative_prob) confidence FROM predictions p LEFT JOIN users u ON u.id=p.user_id {where} ORDER BY p.id DESC LIMIT ?"; a.append(int(limit))
 with closing(_connect()) as c: return pd.read_sql_query(q,c,params=a)
def get_stats(user_id=None):
 where="WHERE user_id=?" if user_id is not None else ""; args=(LABEL_POS,LABEL_NEG)+((int(user_id),) if user_id is not None else ())
 with closing(_connect()) as c: r=c.execute(f"SELECT COUNT(*),SUM(label=?),SUM(label=?),COUNT(DISTINCT batch_id) FROM predictions {where}",args).fetchone()
 return {"total":r[0] or 0,"positive":r[1] or 0,"negative":r[2] or 0,"batch_files":r[3] or 0}
def get_daily_counts(user_id=None):
 where="WHERE user_id=?" if user_id is not None else ""; args=(int(user_id),) if user_id is not None else ()
 with closing(_connect()) as c: return pd.read_sql_query(f"SELECT substr(created_at,1,10) day,label,COUNT(*) n FROM predictions {where} GROUP BY day,label ORDER BY day",c,params=args)
def delete_ids(ids,user_id=None):
 ids=[int(x) for x in ids]; deleted=0
 with closing(_connect()) as c:
  with c:
   for i in range(0,len(ids),500):
    chunk=ids[i:i+500]
    if not chunk: continue
    marks=','.join('?' for _ in chunk); scope=" AND user_id=?" if user_id is not None else ""; args=chunk+([int(user_id)] if user_id is not None else [])
    deleted+=c.execute(f"DELETE FROM predictions WHERE id IN ({marks}){scope}",args).rowcount
 return deleted
def clear_all(user_id=None):
 with closing(_connect()) as c:
  with c: cur=c.execute("DELETE FROM predictions" if user_id is None else "DELETE FROM predictions WHERE user_id=?",() if user_id is None else (int(user_id),)); return cur.rowcount
