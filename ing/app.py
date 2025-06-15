import sys
import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from authlib.integrations.flask_client import OAuth
from spellchecker import SpellChecker
import language_tool_python
import re
import numpy as np # Vektör işlemleri için
import gensim
import spacy

# NLTK importları
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, Laplace # Laplace'ı kullanacağız
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import brown
from nltk.tag import pos_tag
from nltk.metrics.distance import edit_distance

#  (POS Tagger için)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK data (tagger and tokenizer)...")
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('punkt')
    print("Download complete.")

# 1. LANGUAGE TOOL (GRAMER) VE SPELLCHECKER MODELLERİ
tool = language_tool_python.LanguageTool('en-US')
print("LanguageTool initialized.")
spell = SpellChecker()
print("PySpellChecker initialized.")
# 2. FASTTEXT MODELİ
FASTTEXT_MODEL_PATH = '../models/crawl-300d-2M-subword.bin' 
ft_model = None
print(f"Attempting to load FastText model from: {FASTTEXT_MODEL_PATH}")
print("This may take a few minutes on the first run or if the model is large...")
try:
    ft_model = gensim.models.KeyedVectors.load_word2vec_format(FASTTEXT_MODEL_PATH, binary=True)
    
    
    # Modelin yüklenip yüklenmediğini ve temel bir işlem yapıp yapmadığını test et
    if ft_model:
        _ = ft_model['king'] # Test amaçlı bir kelime al
        print(f"FastText model loaded successfully. Vector size: {ft_model.vector_size}")
    else:
        print("FastText model could not be loaded (ft_model is None after load attempt).")
        print("FastText contextual ranking will be disabled.")
except Exception as e:
    print(f"ERROR loading FastText model: {e}")
    print("FastText contextual ranking will be disabled.")
    ft_model = None
# 3. N-GRAM DİL MODELİ (LAPLACE SMOOTHING İLE 0 olması önlenecek)
n_ngram = 3 # Trigram modeli
lm = None   # N-gram modelini global olarak tanımla
print("Attempting to train N-gram model (Laplace)...")
try:
    
    
    # Her cümlenin içindeki her kelimeyi küçük harfe çevir
    training_sents_tokenized = [
        [word.lower() for word in sent] 
        for sent in brown.sents(categories='news')[:2000] 
    ]
    # Eğitim verisini ve kelime dağarcığını hazırla
    train_data, padded_vocab = padded_everygram_pipeline(n_ngram, training_sents_tokenized)
    # Modeli eğit (Laplace smoothing ile)
    lm = Laplace(n_ngram)
    lm.fit(train_data, padded_vocab)
    print(f"{n_ngram}-gram (Laplace) model trained successfully.")
except Exception as e:
    print(f"ERROR training N-gram model: {e}")
    print("N-gram plausibility check will be limited or disabled.")
    lm = None

# 4. SPACY MODELİ
nlp = None
print("Attempting to load spaCy model (en_core_web_sm)...")
try:
    nlp = spacy.load('en_core_web_sm')
    print("spaCy model 'en_core_web_sm' loaded successfully.")
except IOError:
    print("ERROR: spaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
    print("spaCy dependency parsing will be disabled.")
    nlp = None
except Exception as e:
    print(f"ERROR loading spaCy model: {e}")
    print("spaCy dependency parsing will be disabled.")
    nlp = None

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your_secret_key_12345') 
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///../instance/users.db') # Proje kök dizininde instance/users.db
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config['GOOGLE_CLIENT_ID'] = os.environ.get('GOOGLE_CLIENT_ID', 'YOUR_GOOGLE_CLIENT_ID')
app.config['GOOGLE_CLIENT_SECRET'] = os.environ.get('GOOGLE_CLIENT_SECRET', 'YOUR_GOOGLE_CLIENT_SECRET')
app.config['GOOGLE_DISCOVERY_URL'] = "https.accounts.google.com/.well-known/openid-configuration"

db = SQLAlchemy(app)
login_manager = LoginManager(app)
oauth = OAuth(app)
login_manager.login_view = 'login' # Giriş yapılmamışsa yönlendirilecek sayfa
login_manager.login_message_category = 'info' 
oauth.register(
    name='google',
    client_id=app.config['GOOGLE_CLIENT_ID'],
    client_secret=app.config['GOOGLE_CLIENT_SECRET'],
    server_metadata_url=app.config['GOOGLE_DISCOVERY_URL'],
    client_kwargs={
        'scope': 'openid email profile'
    }
)
# --- KULLANICI MODELİ ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256)) # Parola hash'i için daha uzun alan
    google_id = db.Column(db.String(120), unique=True, nullable=True) # Google ile giriş için
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    def __repr__(self):
        return f'<User {self.username}>'
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))
# --- FORMLAR ---
class RegistrationForm(FlaskForm):
    username = StringField('Kullanıcı Adı', 
                           validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('E-posta',
                        validators=[DataRequired(), Email()])
    password = PasswordField('Parola', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Parolayı Doğrula', 
                                     validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Kayıt Ol')
    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Bu kullanıcı adı zaten alınmış. Lütfen farklı bir kullanıcı adı seçin.')
    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Bu e-posta adresi zaten kayıtlı. Lütfen farklı bir e-posta adresi seçin.')
class LoginForm(FlaskForm):
    email = StringField('E-posta',
                        validators=[DataRequired(), Email()])
    password = PasswordField('Parola', validators=[DataRequired()])
    remember = BooleanField('Beni Hatırla')
    submit = SubmitField('Giriş Yap')
# --- YARDIMCI FONKSİYONLAR ---
def generate_candidates(word):
    """
    Bir kelime için olası düzeltme adayları üretir (1 harf değişikliği).
    """
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def tokenize_for_spellchecker(text):
    """Metni yazım denetimi için basitçe kelimelere ayırır ve küçük harfe çevirir."""
    words = re.findall(r'\b\w+\b', text.lower())
    return words

def get_contextual_spell_correction(misspelled_word_lower, candidates, original_sentence_text, ft_model_instance, lm_instance, window_size=2):
    """
    Hibrit bir yaklaşımla (yazım, anlamsal ve frekans) en iyi adayı seçer.
    """
    if not candidates:
        return misspelled_word_lower

    best_candidate = candidates[0]
    max_score = -float('inf')

    # Ağırlıklar
    semantic_weight = 0.4
    frequency_weight = 0.4
    lexical_weight = 0.2

    original_tokens = word_tokenize(original_sentence_text.lower())
    try:
        misspelled_idx = original_tokens.index(misspelled_word_lower)
    except ValueError:
        return candidates[0]

    for candidate_word in candidates:
        # 1. Anlamsal Puan
        semantic_similarity = 0
        if ft_model_instance and candidate_word in ft_model_instance.key_to_index:
            try:
                start = max(0, misspelled_idx - window_size)
                end = min(len(original_tokens), misspelled_idx + window_size + 1)
                context_words = [token for i, token in enumerate(original_tokens) if start <= i < end and i != misspelled_idx and token in ft_model_instance.key_to_index]
                if context_words:
                    context_vector_mean = np.mean([ft_model_instance[word] for word in context_words], axis=0)
                    candidate_vector = ft_model_instance[candidate_word]
                    semantic_similarity = np.dot(candidate_vector, context_vector_mean) / \
                                          (np.linalg.norm(candidate_vector) * np.linalg.norm(context_vector_mean))
            except Exception:
                pass

        # 2. Yazımsal Puan
        lexical_similarity = 1 - (edit_distance(misspelled_word_lower, candidate_word) / max(len(misspelled_word_lower), len(candidate_word)))
        
        # 3. Frekans Puanı
        frequency_score = spell.word_usage_frequency(candidate_word)

        # Hibrit Puan
        hybrid_score = (semantic_similarity * semantic_weight) + (frequency_score * frequency_weight) + (lexical_similarity * lexical_weight)

        if hybrid_score > max_score:
            max_score = hybrid_score
            best_candidate = candidate_word
            
    return best_candidate

def get_dependency_parse(sentence, nlp_instance):
    """
    Analyzes a sentence using spaCy to get dependency parsing information.
    Returns a list of dictionaries, one for each token.
    """
    if not nlp_instance:
        return None
    
    try:
        doc = nlp_instance(sentence)
        parse_info = []
        for token in doc:
            parse_info.append({
                "text": token.text,
                "pos": token.pos_,       # Coarse-grained part-of-speech tag
                "tag": token.tag_,       # Fine-grained part-of-speech tag
                "dep": token.dep_,       # Syntactic dependency relation
                "head_text": token.head.text, # Syntactic head token
                "head_pos": token.head.pos_,  # POS tag of the head token
                "children": [child.text for child in token.children] # List of syntactic children
            })
        return parse_info
    except Exception as e:
        print(f"Error during dependency parsing: {e}")
        return {"error": "Failed to parse sentence.", "details": str(e)}


@app.route('/')
@login_required
def index():
    return render_template('index.html')
# --- KİMLİK DOĞRULAMA ROUTE'LARI ---
@app.route("/kayit", methods=['GET', 'POST'])
def kayit():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data)
        user = User(username=form.username.data, email=form.email.data, password_hash=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Hesabınız başarıyla oluşturuldu! Şimdi giriş yapabilirsiniz.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html', title='Kayıt Ol', form=form)
@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            flash('Başarıyla giriş yaptınız!', 'success')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Giriş başarısız. Lütfen e-posta ve parolanızı kontrol edin.', 'danger')
    return render_template('login.html', title='Giriş Yap', form=form)
@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash('Başarıyla çıkış yaptınız.', 'info')
    return redirect(url_for('login'))
# Google OAuth Route'ları
@app.route('/google/')
def google_login():
    redirect_uri = url_for('google_authorize', _external=True)
    return oauth.google.authorize_redirect(redirect_uri)
@app.route('/google/callback/')
def google_authorize():
    try:
        token = oauth.google.authorize_access_token()
        user_info = oauth.google.parse_id_token(token)
    except Exception as e:
        flash(f'Google ile giriş sırasında bir hata oluştu: {str(e)}', 'danger')
        return redirect(url_for('login'))
    google_id = user_info.get('sub')
    email = user_info.get('email')
   
    profile_name = user_info.get('name')
    
    # Kullanıcıyı veritabanında Google ID ile ara
    user = User.query.filter_by(google_id=google_id).first()
    if not user: # Kullanıcı Google ID ile bulunamadıysa, e-posta ile ara
        user = User.query.filter_by(email=email).first()
        if user: # E-posta ile bulunduysa, Google ID'sini güncelle
            user.google_id = google_id
        else: # E-posta ile de bulunamadıysa, yeni kullanıcı oluştur
            # Basit bir kullanıcı adı oluşturma (e-postanın @ işaretinden önceki kısmı)
            # Daha karmaşık veya kullanıcıya sorma mekanizması eklenebilir
            username_candidate = email.split('@')[0]
            counter = 1
            new_username = username_candidate
            while User.query.filter_by(username=new_username).first():
                new_username = f"{username_candidate}{counter}"
                counter += 1
            
            user = User(
                email=email, 
                username=new_username, # Benzersiz bir kullanıcı adı sağla
                google_id=google_id
                # Parola hash'i Google ile girişte gerekli değil, boş bırakılabilir veya rastgele bir şey atanabilir
            )
        db.session.add(user)
    
    db.session.commit()
    login_user(user)
    flash('Google ile başarıyla giriş yaptınız!', 'success')
    return redirect(url_for('index'))

@app.route('/check_spell', methods=['POST'])
@login_required
def check_spell_route():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        data = request.get_json()
        text_to_check = data.get('text', '')
        if not text_to_check.strip():
            return jsonify({
                "original_text": text_to_check,
                "corrected_text": text_to_check,
                "overall_spell_corrections": [],
                "sentence_analysis": [{"original_sentence": "N/A", "corrected_sentence_for_analysis": "N/A", "spell_corrections_in_sentence": [], "perplexity": {"score": "N/A", "message": "No text provided to check."}}]
            })

        sentences = sent_tokenize(text_to_check)
        all_corrections_made = []
        all_corrected_sentences = []
        sentence_analysis_results = []

        for sentence in sentences:
            # 1. Adım: Yazım denetimi 
            words_original_case = [s for s in re.split(r"([^\w'])", sentence) if s]
            words_for_spellcheck = tokenize_for_spellchecker(sentence)
            misspelled_words = spell.unknown(words_for_spellcheck)
            
            current_sentence_corrections = []
            output_words = list(words_original_case)

            for i, part in enumerate(words_original_case):
                if not re.fullmatch(r'\w+', part):
                    continue
                
                word_lower = part.lower()
                if word_lower in misspelled_words:
                    raw_candidates = list(spell.candidates(word_lower) or [])
                    generated_candidates = generate_candidates(word_lower)
                    all_candidates = set(raw_candidates).union(generated_candidates)
                    known_candidates = list(spell.known(all_candidates))
                    
                    corrected_word = word_lower
                    if not known_candidates:
                        corrected_word = spell.correction(word_lower) or word_lower
                    else:
                        corrected_word = get_contextual_spell_correction(
                            word_lower, known_candidates, sentence, ft_model, lm
                        )

                    if corrected_word and corrected_word.lower() != word_lower:
                        display_corrected = corrected_word
                        if part.istitle():
                            display_corrected = corrected_word.title()
                        elif part.isupper():
                            display_corrected = corrected_word.upper()
                        
                        if display_corrected != part:
                            correction_info = {
                                "original": part,
                                "corrected": display_corrected,
                                "type": "Spelling"
                            }
                            current_sentence_corrections.append(correction_info)
                            all_corrections_made.append(correction_info)
                            output_words[i] = display_corrected
            
            spelling_corrected_sentence = "".join(output_words)

            # 2. Adım: Dilbilgisi denetimi (LanguageTool ile)
            matches = tool.check(spelling_corrected_sentence)
            grammar_corrected_sentence = language_tool_python.utils.correct(spelling_corrected_sentence, matches)

            for match in matches:
                error_text = spelling_corrected_sentence[match.offset:match.offset+match.errorLength]
                corrected_text = match.replacements[0] if match.replacements else error_text
                
                if not any(c['original'] == error_text and c['corrected'] == corrected_text for c in current_sentence_corrections):
                    correction_info = {
                        "original": error_text,
                        "corrected": corrected_text,
                        "type": "Grammar",
                        "message": match.message
                    }
                    current_sentence_corrections.append(correction_info)
                    all_corrections_made.append(correction_info)

            all_corrected_sentences.append(grammar_corrected_sentence)
            corrected_sentence_str = grammar_corrected_sentence

            # Perplexity hesaplaması
            perplexity_score = None
            perplexity_message = "Plausibility check N/A (N-gram model might not be available or sentence unsuitable)."
            if lm:
                try:
                    tokenized_corrected_sentence = word_tokenize(corrected_sentence_str.lower())
                    if tokenized_corrected_sentence:
                        filtered_tokens = [token for token in tokenized_corrected_sentence if token in lm.vocab]
                        if len(filtered_tokens) >= n_ngram:
                            temp_score = lm.perplexity(filtered_tokens)
                            if temp_score == float('inf') or temp_score != temp_score:
                                perplexity_message = "This sentence structure is highly unusual (Perplexity: Undefined)."
                            else:
                                perplexity_score = temp_score
                                if perplexity_score > 1000:
                                    perplexity_message = f"This sentence might be unusual (Perplexity: {perplexity_score:.2f})."
                                else:
                                    perplexity_message = f"Sentence seems plausible (Perplexity: {perplexity_score:.2f})."
                        else:
                            perplexity_message = "Sentence too short or has too many unrecognized words for a reliable perplexity calculation."
                    else:
                        perplexity_message = "Cannot calculate perplexity for an empty sentence."
                except Exception as e_perplexity:
                    print(f"ERROR calculating perplexity: {e_perplexity}")
                    perplexity_message = "An error occurred during perplexity calculation."
            
            # Dependency Parse analizi
            dependency_parse_result = get_dependency_parse(corrected_sentence_str, nlp)

            sentence_analysis_results.append({
                "original_sentence": sentence,
                "corrected_sentence_for_analysis": corrected_sentence_str,
                "spell_corrections_in_sentence": current_sentence_corrections,
                "perplexity": {
                    "score": f"{perplexity_score:.2f}" if perplexity_score is not None else "N/A",
                    "message": perplexity_message
                },
                "dependency_parse": dependency_parse_result or []
            })

        final_corrected_text = " ".join(all_corrected_sentences)
        return jsonify({
            "original_text": text_to_check,
            "corrected_text": final_corrected_text,
            "overall_spell_corrections": all_corrections_made,
            "sentence_analysis": sentence_analysis_results
        })
    except Exception as e: 
        print("!!!!!!!!!!!! OVERALL EXCEPTION IN CHECK_SPELL_ROUTE !!!!!!!!!!!!!")
        import traceback
        traceback.print_exc() 
        return jsonify({"error": "An internal server error occurred on the backend.", "details": str(e)}), 500

# --- UYGULAMAYI ÇALIŞTIRMA ---
if __name__ == '__main__':
    with app.app_context():
       
        instance_path = os.path.join(app.root_path, '../instance')
        if not os.path.exists(instance_path):
            try:
                os.makedirs(instance_path)
                print(f"Created instance folder at: {instance_path}")
            except OSError as e:
                print(f"Error creating instance folder {instance_path}: {e}")
        
        db.create_all()
        print(f"Database tables created (if they didn't exist) at {app.config['SQLALCHEMY_DATABASE_URI']}")
   
    app.run(debug=True)
