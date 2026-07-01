import os
import json
import hashlib
import re
import requests
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from ocr_module import extract_text_from_file
from rag_module import find_similar_normative

# --- Настройки приложения ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey_change_this_in_production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:pass@localhost:5433/intel_gateway'
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- Модели базы данных (создадутся в PostgreSQL) ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), default='operator')

    def set_password(self, password):
        self.password_hash = hashlib.sha256(password.encode()).hexdigest()

    def check_password(self, password):
        return self.password_hash == hashlib.sha256(password.encode()).hexdigest()

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255))
    file_hash = db.Column(db.String(32), unique=True)
    text_hash = db.Column(db.String(32))
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    ocr_text = db.Column(db.Text)
    summary = db.Column(db.String(500))
    doc_type = db.Column(db.String(50))
    urgency = db.Column(db.Boolean, default=False)
    control_date = db.Column(db.DateTime)
    assigned_department = db.Column(db.String(100))
    sender_organization = db.Column(db.String(200))
    sender_name = db.Column(db.String(200))
    extracted_entities = db.Column(db.Text)
    status = db.Column(db.String(20), default='new')
    confirmed = db.Column(db.Boolean, default=False)

    @property
    def doc_type_ru(self):
        return {'request': 'Запрос', 'citizen_appeal': 'Обращение гражданина', 
                'notification': 'Уведомление', 'other': 'Прочее'}.get(self.doc_type, self.doc_type)

    @property
    def status_ru(self):
        return {'new': 'Новый', 'confirmed': 'Подтверждён', 'overdue': 'Просрочен'}.get(self.status, self.status)

class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    action = db.Column(db.String(255))
    document_id = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# --- Вспомогательные функции ---
def compute_file_hash(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def compute_text_hash(text):
    return hashlib.md5(normalize_text(text).encode('utf-8')).hexdigest()

def extract_sender_from_text(text):
    org = ""
    name = ""
    lines = text.split('\n')
    for line in lines[:5]:
        if len(line.strip()) > 5 and any(kw in line.lower() for kw in ['увд','гувд','овд','прокуратура','министерство','департамент','управление']):
            org = line.strip()
            break
    if not org:
        m = re.search(r'([А-ЯЁ\s]{5,})?(?:УВД|ГУВД|ОВД|ПРОКУРАТУРА|МИНИСТЕРСТВО|ДЕПАРТАМЕНТ)[А-ЯЁ\s]{5,}', text, re.IGNORECASE)
        if m:
            org = m.group(1).strip()

    m = re.search(r'(?:Начальник|Директор|Главный|Исполнитель|Начальник МОБ)\s+([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)', text)
    if m:
        name = m.group(1)
    else:
        m = re.search(r'([А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.[А-ЯЁ]\.)', text)
        if m:
            name = m.group(1)
    return org, name

# --- LLM (Ollama) ---
def call_llm_for_classification(text, context):
    ollama_host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
    prompt = f"""Ты – эксперт по делопроизводству. Верни только JSON. Контекст: {context}
Текст: {text[:3000]}
Формат: {{"doc_type": "citizen_appeal|request|notification|other", "urgency": true/false, "control_date": "YYYY-MM-DD" или null, "assigned_department": "один из: Правовой департамент, Отдел по работе с обращениями граждан, Департамент цифрового развития, Департамент экономики и финансов, Секретариат, Административный департамент, Отдел межведомственного взаимодействия", "entities": {{"name": "ФИО отправителя", "number": "номер", "subject": "суть", "organization": "организация отправителя"}} }}"""
    try:
        r = requests.post(f'{ollama_host}/api/generate', json={'model': 'qwen2.5:3b', 'prompt': prompt, 'stream': False}, timeout=90)
        result = r.json()['response']
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1:
            return result[start:end]
    except Exception as e:
        print(f"Error calling LLM: {e}")
        pass
    return '{"doc_type":"other","urgency":false,"control_date":null,"assigned_department":"Канцелярия","entities":{}}'

def generate_summary(text):
    ollama_host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
    prompt = f'Краткая суть (макс 15 слов):\n{text[:1500]}\nСуть:'
    try:
        r = requests.post(f'{ollama_host}/api/generate', json={'model': 'qwen2.5:3b', 'prompt': prompt, 'stream': False}, timeout=60)
        s = r.json()['response'].strip()
        return s[:500] + ('...' if len(s) > 500 else "")
    except:
        return "Краткое описание не сформировано"

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# --- Маршруты ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and user.check_password(request.form['password']):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Неверный логин или пароль', 'danger')
    return '''
    <form method="post" style="width: 300px; margin: 50px auto; text-align: center;">
        <h3>Вход в систему</h3>
        <p>Логин: <input type="text" name="username" required></p>
        <p>Пароль: <input type="password" name="password" required></p>
        <p><input type="submit" value="Войти"></p>
    </form>
    '''

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def dashboard():
    docs = Document.query.order_by(Document.upload_date.desc()).limit(50).all()
    overdue = Document.query.filter(Document.control_date < datetime.utcnow(), Document.status != 'overdue').count()
    return render_template('dashboard.html', documents=docs, overdue_count=overdue)

@app.route('/upload', methods=['POST'])
@login_required
def upload_document():
    file = request.files['file']
    if not file:
        flash('Файл не выбран', 'danger')
        return redirect(url_for('dashboard'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    file_hash = compute_file_hash(filepath)
    existing = Document.query.filter_by(file_hash=file_hash).first()
    if existing:
        flash(f'Файл уже загружен. Карточка №{existing.id}', 'warning')
        return redirect(url_for('document_card', doc_id=existing.id))

    text = extract_text_from_file(filepath)
    if not text or len(text) < 20:
        flash('Не удалось распознать текст (используйте чёткое изображение).', 'danger')
        return redirect(url_for('dashboard'))

    text_hash = compute_text_hash(text)
    existing = Document.query.filter_by(text_hash=text_hash).first()
    if existing:
        flash(f'Документ с таким же содержанием уже существует. Карточка №{existing.id}', 'warning')
        return redirect(url_for('document_card', doc_id=existing.id))

    context = find_similar_normative(text)
    json_response = call_llm_for_classification(text, context)
    
    try:
        data = json.loads(json_response)
    except json.JSONDecodeError:
        data = {"doc_type": "other", "urgency": False, "assigned_department": "Канцелярия"}

    # Жёсткие правила
    text_lower = text.lower()
    if re.search(r'№\s*23/5076', text) or re.search(r'начальник\s+моб', text_lower) or any(kw in text_lower for kw in ['мвд','гувд','прокуратура','кусп','материал','проверк','милиции']):
        data['doc_type'] = 'request'
        data['urgency'] = True
        data['assigned_department'] = 'Правовой департамент'

    if any(kw in text_lower for kw in ['обращение','жалоба','прошу','гражданин']):
        if data.get('doc_type') != 'request':
            data['doc_type'] = 'citizen_appeal'
            data['assigned_department'] = 'Отдел по работе с обращениями граждан'

    if any(kw in text_lower for kw in ['срочно','короткий срок','незамедлител']):
        data['urgency'] = True

    control_date = None
    if data.get('urgency'):
        control_date = datetime.utcnow() + timedelta(days=3)
    elif data.get('control_date'):
        try:
            control_date = datetime.strptime(data['control_date'], '%Y-%m-%d')
        except:
            pass
    elif data.get('doc_type') == 'citizen_appeal':
        control_date = datetime.utcnow() + timedelta(days=30)
    elif data.get('doc_type') == 'request':
        control_date = datetime.utcnow() + timedelta(days=15)

    summary = generate_summary(text)
    entities = data.get('entities', {})
    org, name = entities.get('organization', ''), entities.get('name', '')
    if not org or not name:
        ro, rn = extract_sender_from_text(text)
        if not org:
            org = ro
        if not name:
            name = rn

    doc = Document(
        filename=filename,
        file_hash=file_hash,
        text_hash=text_hash,
        ocr_text=text,
        summary=summary,
        doc_type=data.get('doc_type', 'other'),
        urgency=data.get('urgency', False),
        control_date=control_date,
        assigned_department=data.get('assigned_department', 'Канцелярия'),
        sender_organization=org,
        sender_name=name,
        extracted_entities=json.dumps(entities),
        status='new'
    )
    db.session.add(doc)
    db.session.commit()
    db.session.add(AuditLog(user_id=current_user.id, action='upload', document_id=doc.id))
    db.session.commit()

    flash(f'Документ загружен. Карточка №{doc.id}', 'success')
    return redirect(url_for('document_card', doc_id=doc.id))

@app.route('/doc/<int:doc_id>')
@login_required
def document_card(doc_id):
    doc = Document.query.get_or_404(doc_id)
    entities = json.loads(doc.extracted_entities) if doc.extracted_entities else {}
    return render_template('document_card.html', doc=doc, entities=entities)

@app.route('/doc/<int:doc_id>/confirm', methods=['POST'])
@login_required
def confirm_document(doc_id):
    doc = Document.query.get_or_404(doc_id)
    doc.status = 'confirmed'
    doc.confirmed = True
    db.session.commit()
    
    print(f"[ЭМУЛЯТОР СЭД] Отправлен документ {doc.id}: тип={doc.doc_type}, срок={doc.control_date}, отдел={doc.assigned_department}")
    flash(f'Документ №{doc.id} передан в СЭД', 'success')
    
    db.session.add(AuditLog(user_id=current_user.id, action='confirm', document_id=doc.id))
    db.session.commit()
    return redirect(url_for('dashboard'))

@app.route('/doc/<int:doc_id>/delete', methods=['POST'])
@login_required
def delete_document(doc_id):
    doc = Document.query.get_or_404(doc_id)
    AuditLog.query.filter_by(document_id=doc_id).delete()
    db.session.delete(doc)
    db.session.commit()
    flash(f'Документ №{doc_id} удалён.', 'success')
    return redirect(url_for('dashboard'))

@app.route('/control')
@login_required
def control_panel():
    docs = Document.query.filter(Document.control_date != None).order_by(Document.control_date).all()
    now = datetime.utcnow()
    return render_template('control_panel.html', documents=docs, now=now)

@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Добавление новых колонок, если их ещё нет (для PostgreSQL синтаксис ALTER TABLE тоже сработает)
        for col in ['file_hash', 'text_hash', 'sender_organization', 'sender_name']:
            try:
                db.session.execute(f'ALTER TABLE document ADD COLUMN {col} VARCHAR(200);')
                db.session.commit()
            except:
                pass
        if not User.query.filter_by(username='admin').first():
            admin = User(username='admin', role='admin')
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
    app.run(debug=True, host='0.0.0.0', port=5000)