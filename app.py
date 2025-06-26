from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import pandas as pd
import numpy as np
import requests
import difflib
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import base64
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import paypalrestsdk

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your-secret-key-here'  # Change this to a random secret key

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('pending_jobs', exist_ok=True)

# Enhanced stop words
english = ENGLISH_STOP_WORDS
extra = {
    'customer', 'client', 'user', 'account', 'admin', 'app', 'dashboard',
    'click', 'issue', 'feature', 'time', 'day', 'week', 'month', 'year',
    'good', 'bad', 'new', 'old', 'big', 'small', 'high', 'low', 'said', 'say'
}
all_stop = set(english).union(extra)

# Configure PayPal SDK
paypal_client_id = os.environ.get('PAYPAL_CLIENT_ID', '')
paypal_client_secret = os.environ.get('PAYPAL_CLIENT_SECRET', '')

# Debug PayPal credentials (remove after testing)
print(f"PayPal Client ID present: {bool(paypal_client_id)}")
print(f"PayPal Client Secret present: {bool(paypal_client_secret)}")
if paypal_client_id:
    print(f"Client ID starts with: {paypal_client_id[:10]}...")

# Check if PayPal credentials are available
PAYPAL_ENABLED = bool(paypal_client_id and paypal_client_secret)

if PAYPAL_ENABLED:
    paypalrestsdk.configure({
        "mode": "sandbox",  # Change to "live" for production
        "client_id": paypal_client_id,
        "client_secret": paypal_client_secret
    })
    print("PayPal SDK configured successfully")
else:
    print("PayPal credentials not available - payment disabled")

# Global variables
current_data = None
clustering_result = None

# Service tiers configuration
SERVICE_TIERS = {
    'free': {
        'name': 'DIAM Analysis',
        'price': 0,
        'max_file_size': 100,  # MB
        'description': 'Advanced clustering analysis with all features included',
        'features': [
            'Up to 100MB file size',
            'Advanced cluster analysis',
            'AI-generated strategic insights',
            'Cross-variable analysis',
            'Visual charts and summaries',
            'CSV export of results',
            'Instant real-time processing',
            'Strategic AI prompt generation'
        ]
    },
    'enterprise': {
        'name': 'Enterprise Consultation',
        'price': 'Custom',
        'max_file_size': 'Unlimited',  # MB
        'description': 'Complete business transformation with personal consultation',
        'features': [
            'Everything in DIAM Analysis',
            'Unlimited file size processing',
            'Personal consultation call',
            'Custom analysis framework',
            'Implementation roadmap',
            'Follow-up support',
            '30-day business transformation plan',
            'Priority email and phone support'
        ]
    }
}

def validate_job_id(job_id):
    """Validate job_id to prevent path traversal attacks"""
    import re
    # Only allow alphanumeric characters and hyphens (UUID format)
    if not isinstance(job_id, str):
        return False
    if not re.match(r'^[a-f0-9-]{36}$', job_id):
        return False
    # Ensure no path traversal characters
    if '..' in job_id or '/' in job_id or '\\' in job_id:
        return False
    return True

def safe_job_file_path(job_id):
    """Safely construct job file path after validation"""
    if not validate_job_id(job_id):
        raise ValueError("Invalid job ID format")
    return os.path.join('pending_jobs', f'{job_id}.json')

def save_job_request(job_data):
    """Save job request to pending jobs directory"""
    job_id = str(uuid.uuid4())
    job_data['job_id'] = job_id
    job_data['timestamp'] = datetime.now().isoformat()

    file_path = safe_job_file_path(job_id)
    with open(file_path, 'w') as f:
        json.dump(job_data, f, indent=2)

    return job_id



def save_to_database(job_data, file_info=None):
    """Save job data to Replit database as alternative to Google Sheets"""
    try:
        import os
        import requests
        import json
        from datetime import datetime

        db_url = os.getenv("REPLIT_DB_URL")
        if not db_url:
            print("Database not available")
            return False

        # Prepare the data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Store individual job data
        job_key = f"job_{job_data.get('job_id', '')}"
        job_record = {
            'timestamp': timestamp,
            'job_id': job_data.get('job_id', ''),
            'tier': job_data.get('tier', ''),
            'full_name': job_data.get('full_name', ''),
            'company_name': job_data.get('company_name', ''),
            'email': job_data.get('email', ''),
            'industry': job_data.get('industry', ''),
            'company_size': job_data.get('company_size', ''),
            'timeline': job_data.get('timeline', ''),
            'budget_range': job_data.get('budget_range', ''),
            'requirements': job_data.get('requirements', ''),
            'status': job_data.get('status', ''),
            'price': str(job_data.get('price', '')),
            'file_name': file_info.get('filename', '') if file_info else '',
            'file_size_mb': str(round(file_info.get('size_mb', 0), 2)) if file_info else '',
            'file_path': file_info.get('path', '') if file_info else '',
            'processing_status': 'Pending' if file_info else 'No File'
        }

        # Save job record
        response = requests.post(f"{db_url}/{job_key}", data=json.dumps(job_record))

        # Also add to jobs list for easy retrieval
        list_key = "all_jobs"
        try:
            existing_jobs = requests.get(f"{db_url}/{list_key}")
            if existing_jobs.status_code == 200:
                jobs_list = json.loads(existing_jobs.text)
            else:
                jobs_list = []
        except:
            jobs_list = []

        jobs_list.append(job_data.get('job_id', ''))
        requests.post(f"{db_url}/{list_key}", data=json.dumps(jobs_list))

        print(f"Successfully saved job data to database for job {job_data.get('job_id')}")
        return True

    except Exception as e:
        print(f"Error saving to database: {str(e)}")
        return False

def log_to_google_sheets(job_data, file_info=None):
    """Log user submission and file data to database only"""
    return save_to_database(job_data, file_info)

def create_paypal_payment(amount, description="Professional Data Analysis"):
    """Create PayPal payment"""
    try:
        if not PAYPAL_ENABLED:
            print("PayPal credentials not configured")
            return None
        
        print(f"Creating PayPal payment for amount: ${amount}")
        
        payment = paypalrestsdk.Payment({
            "intent": "sale",
            "payer": {
                "payment_method": "paypal"
            },
            "redirect_urls": {
                "return_url": f"{request.url_root}payment/success",
                "cancel_url": f"{request.url_root}payment/cancel"
            },
            "transactions": [{
                "item_list": {
                    "items": [{
                        "name": description,
                        "sku": "data-analysis",
                        "price": str(amount),
                        "currency": "USD",
                        "quantity": 1
                    }]
                },
                "amount": {
                    "total": str(amount),
                    "currency": "USD"
                },
                "description": description
            }]
        })

        if payment.create():
            print(f"PayPal payment created successfully: {payment.id}")
            return payment
        else:
            print(f"PayPal payment creation failed: {payment.error}")
            return None
    except Exception as e:
        print(f"PayPal payment exception: {str(e)}")
        return None

def execute_paypal_payment(payment_id, payer_id):
    """Execute PayPal payment"""
    try:
        payment = paypalrestsdk.Payment.find(payment_id)
        if payment.execute({"payer_id": payer_id}):
            return True
        else:
            print(f"PayPal execution error: {payment.error}")
            return False
    except Exception as e:
        print(f"PayPal execution error: {str(e)}")
        return False

def send_confirmation_email(email, name, job_id, tier):
    """Send confirmation email (placeholder - implement with your email service)"""
    # This is a placeholder - you'll need to configure with your email service
    print(f"Sending confirmation email to {email} for job {job_id} ({tier} tier)")
    return True

# Include all the previous clustering functions (keeping them exactly as they were)
def download_from_drive(file_id: str) -> bytes:
    """Download file from Google Drive with enhanced support for large files"""
    import tempfile
    import os

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    sess = requests.Session()
    sess.headers.update(headers)

    urls_to_try = [
        f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t",
        f"https://docs.google.com/uc?export=download&id={file_id}&confirm=t",
        f"https://drive.google.com/uc?export=download&id={file_id}",
        f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv&gid=0",
        f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx"
    ]

    for i, url in enumerate(urls_to_try):
        try:
            r = sess.get(url, stream=True, allow_redirects=True, timeout=300)

            if 'download_warning' in r.cookies or 'download_warning' in r.text:
                token = None
                for cookie_name, cookie_value in r.cookies.items():
                    if 'download_warning' in cookie_name:
                        token = cookie_value
                        break

                if not token and 'confirm=' in r.text:
                    import re
                    match = re.search(r'confirm=([^&"]+)', r.text)
                    if match:
                        token = match.group(1)

                if token:
                    params = {'id': file_id, 'confirm': token, 'export': 'download'}
                    r = sess.get("https://docs.google.com/uc", params=params, stream=True, timeout=300)

            content_chunks = []
            total_size = 0
            max_size = 100 * 1024 * 1024  # 100MB limit

            try:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        content_chunks.append(chunk)
                        total_size += len(chunk)

                        if total_size > max_size:
                            return b"File too large"

                content = b''.join(content_chunks)

            except Exception as e:
                continue

            if len(content) < 100:
                continue

            content_start = content[:1000].lower()
            if (b'<!doctype html' in content_start or 
                b'<html' in content_start or
                b'google drive' in content_start or
                b'access denied' in content_start):
                continue

            return content

        except Exception as e:
            continue

    return b"All download methods failed"

def preprocess_text(text_series):
    return (text_series
            .fillna("")
            .astype(str)
            .str.lower()
            .str.replace(r'[^\w\s]', ' ', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip())

def compute_advanced_clusters(data, column_info, k, max_feat=500, ngram_range=(1,2), optimize=False):
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    import time

    start_time = time.time()
    data_type = column_info['type']
    raw_data = column_info['data']
    column_name = column_info['name']

    sampled_data = raw_data
    sample_indices = list(range(len(raw_data)))
    is_sampled = False

    if data_type == 'text':
        processed_texts = preprocess_text(pd.Series(sampled_data)).tolist()
        non_empty_indices = [i for i, text in enumerate(processed_texts) if text.strip()]
        filtered_texts = [processed_texts[i] for i in non_empty_indices]

        if len(filtered_texts) < k:
            k = max(2, len(filtered_texts) // 2)

        if len(filtered_texts) > 20000:
            max_feat = min(max_feat, 200)
        elif len(filtered_texts) > 10000:
            max_feat = min(max_feat, 300)

        vect = TfidfVectorizer(
            max_features=max_feat,
            ngram_range=ngram_range,
            stop_words=list(all_stop),
            min_df=max(2, len(filtered_texts) // 1000),
            max_df=0.95
        )
        X = vect.fit_transform(filtered_texts)
        feature_names = vect.get_feature_names_out().tolist()

    elif data_type == 'numeric':
        df_clean = pd.DataFrame({column_name: sampled_data}).dropna()
        non_empty_indices = df_clean.index.tolist()

        if len(df_clean) < k:
            k = max(2, len(df_clean) // 2)

        scaler = StandardScaler()
        X = scaler.fit_transform(df_clean.values.reshape(-1, 1))
        feature_names = [f"{column_name}_scaled"]
        processed_texts = df_clean[column_name].astype(str).tolist()

    elif data_type == 'categorical':
        df_clean = pd.DataFrame({column_name: sampled_data}).dropna()
        non_empty_indices = df_clean.index.tolist()

        if len(df_clean) < k:
            k = max(2, len(df_clean) // 2)

        value_counts = df_clean[column_name].value_counts()
        if len(value_counts) > 100:
            top_categories = value_counts.head(100).index
            df_clean[column_name] = df_clean[column_name].where(
                df_clean[column_name].isin(top_categories), 'Other'
            )

        dummies = pd.get_dummies(df_clean[column_name], prefix=column_name)
        X = dummies.values
        feature_names = dummies.columns.tolist()
        processed_texts = df_clean[column_name].astype(str).tolist()

    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    best_k = k
    silhouette_scores = {}

    if optimize and len(non_empty_indices) >= 6:
        max_k_to_test = min(12, len(non_empty_indices) // 10)
        k_range = range(3, max_k_to_test)

        for test_k in k_range:
            km_test = KMeans(n_clusters=test_k, random_state=42, n_init=5, max_iter=100)
            labels_test = km_test.fit_predict(X)

            if X.shape[0] > 3000:
                score = -km_test.inertia_ / X.shape[0]
            else:
                score = silhouette_score(X, labels_test)

            silhouette_scores[test_k] = score

            if time.time() - start_time > 60:
                break

        if silhouette_scores:
            best_k = max(silhouette_scores, key=silhouette_scores.get)

    km = KMeans(n_clusters=best_k, random_state=42, n_init=5, max_iter=200)
    labels = km.fit_predict(X)

    full_labels = np.full(len(raw_data), -1)

    if is_sampled:
        for i, orig_sample_idx in enumerate(sample_indices):
            if i < len(non_empty_indices):
                local_idx = non_empty_indices[i]
                if local_idx < len(labels):
                    full_labels[orig_sample_idx] = labels[local_idx]
    else:
        for i, orig_idx in enumerate(non_empty_indices):
            if i < len(labels):
                full_labels[orig_idx] = labels[i]

    return {
        "data": raw_data,
        "processed_data": processed_texts if data_type == 'text' else [str(x) for x in sampled_data],
        "labels": full_labels.tolist(),
        "features": feature_names,
        "centers": km.cluster_centers_.tolist(),
        "mean_center": km.cluster_centers_.mean(axis=0).tolist(),
        "k": best_k,
        "silhouette_scores": silhouette_scores,
        "non_empty_indices": non_empty_indices,
        "data_type": data_type,
        "column_name": column_name,
        "is_sampled": is_sampled,
        "sample_size": len(sampled_data),
        "total_records": len(raw_data)
    }

def get_enhanced_influencers(result, top_n=25):
    influencers = {}
    cluster_stats = {}
    data_type = result.get("data_type", "text")

    for cid in range(result["k"]):
        cluster_data = [result["data"][i] for i, label in enumerate(result["labels"]) if label == cid]
        cluster_stats[cid] = {
            "count": len(cluster_data),
            "avg_length": np.mean([len(str(text)) for text in cluster_data]) if cluster_data else 0
        }

        if data_type == 'text':
            if cid < len(result["centers"]):
                diffs = np.array(result["centers"][cid]) - np.array(result["mean_center"])
                top_indices = np.argsort(diffs)[::-1][:top_n*4]

                words = []
                phrases = []

                for idx in top_indices:
                    feature = result["features"][idx]

                    if ' ' in feature:
                        if (not any(ch.isdigit() for ch in feature) and 
                            len(feature) > 3 and
                            not any(stop_word in feature.lower() for stop_word in ['the ', ' the', 'and ', ' and', 'of ', ' of'])):
                            if not any(difflib.SequenceMatcher(None, feature, existing).ratio() > 0.7 
                                      for existing in phrases):
                                phrases.append(feature)
                                if len(phrases) >= top_n//2:
                                    break
                    else:
                        if (not any(ch.isdigit() for ch in feature) and 
                            feature not in all_stop and 
                            len(feature) > 2):
                            if not any(difflib.SequenceMatcher(None, feature, existing).ratio() > 0.8 
                                      for existing in words):
                                words.append(feature)
                                if len(words) >= top_n*2:
                                    break

                combined_keywords = phrases + words[:top_n-len(phrases)]
                influencers[cid] = combined_keywords[:top_n]

        elif data_type == 'numeric':
            if cluster_data:
                cluster_values = [float(x) for x in cluster_data if pd.notna(x)]
                if cluster_values:
                    q25, q75 = np.percentile(cluster_values, [25, 75])
                    influencers[cid] = [
                        f"Range: {min(cluster_values):.2f} - {max(cluster_values):.2f}",
                        f"Mean: {np.mean(cluster_values):.2f}",
                        f"Median: {np.median(cluster_values):.2f}",
                        f"Q1-Q3: {q25:.2f} - {q75:.2f}",
                        f"Std Dev: {np.std(cluster_values):.2f}"
                    ]
                else:
                    influencers[cid] = ["No valid numeric values"]
            else:
                influencers[cid] = ["Empty cluster"]

        elif data_type == 'categorical':
            if cluster_data:
                value_counts = pd.Series(cluster_data).value_counts()
                total_count = len(cluster_data)
                top_categories = value_counts.head(top_n).index.tolist()
                influencers[cid] = [f"{cat} ({value_counts[cat]}, {value_counts[cat]/total_count*100:.1f}%)" 
                                  for cat in top_categories]
            else:
                influencers[cid] = ["Empty cluster"]

        else:
            influencers[cid] = ["Unsupported data type"]

    return influencers, cluster_stats

@app.route('/')
def index():
    return render_template('index.html', service_tiers=SERVICE_TIERS)

@app.route('/analyzer')
def analyzer():
    # Check if this is a tier-specific request
    tier = request.args.get('tier', None)
    if tier:
        # For specific tiers, render the analyzer with the tier pre-selected
        return render_template('index.html', service_tiers=SERVICE_TIERS, selected_tier=tier)
    return render_template('index.html', service_tiers=SERVICE_TIERS)

@app.route('/consultancy')
def consultancy():
    return render_template('consultancy.html', service_tiers=SERVICE_TIERS)

@app.route('/create_payment', methods=['POST'])
def create_payment():
    """Create PayPal payment"""
    try:
        if not PAYPAL_ENABLED:
            return jsonify({
                'success': False, 
                'message': 'Payment system is currently unavailable. Please contact support.'
            })
            
        data = request.get_json()
        amount = data.get('amount', 50)

        payment = create_paypal_payment(amount)
        if payment:
            # Store payment info temporarily
            payment_info = {
                'payment_id': payment.id,
                'amount': amount,
                'timestamp': datetime.now().isoformat()
            }

            # Save payment info
            payment_file = f"pending_jobs/payment_{payment.id}.json"
            with open(payment_file, 'w') as f:
                json.dump(payment_info, f, indent=2)

            # Get approval URL
            approval_url = None
            for link in payment.links:
                if link.rel == "approval_url":
                    approval_url = link.href
                    break

            return jsonify({
                'success': True,
                'payment_id': payment.id,
                'approval_url': approval_url
            })
        else:
            return jsonify({
                'success': False, 
                'message': 'Unable to process payment at this time. Please try again later.'
            })

    except Exception as e:
        return jsonify({'success': False, 'message': f'Payment processing error. Please try again.'})

@app.route('/payment/success')
def payment_success():
    """Handle successful PayPal payment"""
    try:
        payment_id = request.args.get('paymentId')
        payer_id = request.args.get('PayerID')

        if execute_paypal_payment(payment_id, payer_id):
            # Payment successful - redirect to main analyzer with professional features
            return redirect('/?payment_verified=true&tier=professional')
        else:
            return render_template('payment_error.html', message='Payment execution failed')

    except Exception as e:
        return render_template('payment_error.html', message=f'Payment error: {str(e)}')

@app.route('/payment/cancel')
def payment_cancel():
    """Handle cancelled PayPal payment"""
    return redirect('/?payment_cancelled=true')

@app.route('/submit_job', methods=['POST'])
def submit_job():
    try:
        data = request.get_json()
        tier = data.get('tier')

        if tier not in SERVICE_TIERS:
            return jsonify({'success': False, 'message': 'Invalid service tier'})

        # Create job request
        job_data = {
            'tier': tier,
            'full_name': data.get('full_name'),
            'company_name': data.get('company_name'),
            'email': data.get('email'),
            'status': 'pending_payment' if tier == 'professional' else 'pending_file',
            'price': SERVICE_TIERS[tier]['price']
        }

        # Handle enterprise tier
        if tier == 'enterprise':
            job_data.update({
                'industry': data.get('industry'),
                'company_size': data.get('company_size'),
                'timeline': data.get('timeline'),
                'budget_range': data.get('budget_range'),
                'requirements': data.get('requirements'),
                'status': 'pending_review'
            })

        job_id = save_job_request(job_data)

        # Log to Google Sheets
        log_to_google_sheets(job_data)

        if tier == 'free':
            return jsonify({
                'success': True,
                'message': 'Free analysis request submitted!',
                'redirect': f'/upload/{job_id}',
                'job_id': job_id
            })
        else:  # enterprise
            return jsonify({
                'success': True,
                'message': 'Enterprise consultation request submitted! We will contact you within 24 hours.',
                'job_id': job_id
            })

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error submitting request: {str(e)}'})

@app.route('/upload/<job_id>')
def upload_page(job_id):
    # Load job data with security validation
    try:
        if not validate_job_id(job_id):
            return "Invalid job ID", 400

        file_path = safe_job_file_path(job_id)
        with open(file_path, 'r') as f:
            job_data = json.load(f)

        tier_info = SERVICE_TIERS[job_data['tier']]
        return render_template('upload.html', job_data=job_data, tier_info=tier_info, job_id=job_id)
    except ValueError as e:
        return str(e), 400
    except FileNotFoundError:
        return "Job not found", 404
    except Exception as e:
        return "Error loading job", 500

@app.route('/upload_file/<job_id>', methods=['POST'])
def upload_file_for_job(job_id):
    try:
        # Validate job_id for security
        if not validate_job_id(job_id):
            return jsonify({'success': False, 'message': 'Invalid job ID format'})

        # Load job data
        file_path = safe_job_file_path(job_id)
        with open(file_path, 'r') as f:
            job_data = json.load(f)

        tier_info = SERVICE_TIERS[job_data['tier']]
        max_size = tier_info['max_file_size'] * 1024 * 1024  # Convert MB to bytes

        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})

        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        if file_size > max_size:
            return jsonify({
                'success': False, 
                'message': f'File too large. Maximum size for {tier_info["name"]} is {tier_info["max_file_size"]}MB'
            })

        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
        file.save(file_path)

        # Update job data
        job_data['file_path'] = file_path
        job_data['original_filename'] = filename
        job_data['file_size'] = file_size
        job_data['status'] = 'file_uploaded'

        file_path = safe_job_file_path(job_id)
        with open(file_path, 'w') as f:
            json.dump(job_data, f, indent=2)

        # Log file info to Google Sheets
        file_info = {
            'filename': filename,
            'size_mb': file_size / (1024 * 1024),
            'path': file_path
        }
        log_to_google_sheets(job_data, file_info)

        # Send confirmation email
        send_confirmation_email(job_data['email'], job_data['full_name'], job_id, job_data['tier'])

        if job_data['tier'] == 'free':
            return jsonify({
                'success': True,
                'message': 'File uploaded successfully! Processing your free analysis...',
                'redirect': f'/process_realtime/{job_id}'
            })
        elif job_data['tier'] == 'professional':
            return jsonify({
                'success': True,
                'message': 'File uploaded successfully! Processing your professional analysis...',
                'redirect': f'/process_realtime/{job_id}'
            })
        else:  # enterprise
            return jsonify({
                'success': True,
                'message': 'File uploaded successfully! Your enterprise analysis will be completed within 1-2 hours and sent via email.'
            })

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error uploading file: {str(e)}'})

def validate_db_key(key):
    """Validate database key to prevent injection attacks"""
    import re
    # Only allow alphanumeric characters, hyphens, and underscores
    if not isinstance(key, str) or not re.match(r'^[a-zA-Z0-9_-]+$', key):
        return False
    # Prevent path traversal
    if '..' in key or '/' in key or '\\' in key:
        return False
    # Limit key length
    if len(key) > 100:
        return False
    return True

def validate_db_url(url):
    """Validate that the URL is the official Replit database URL"""
    if not url:
        return False
    # Only allow Replit database URLs
    allowed_domains = ['kv.replit.com', 'database.replit.com']
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return parsed.hostname in allowed_domains and parsed.scheme == 'https'

@app.route('/api/db/<key>', methods=['GET', 'POST', 'DELETE'])
def database_api(key):
    """API endpoint to access Replit database with security validations"""
    try:
        import os
        import requests

        # Validate the database key
        if not validate_db_key(key):
            return jsonify({'error': 'Invalid key format'}), 400

        db_url = os.getenv("REPLIT_DB_URL")
        if not db_url:
            return jsonify({'error': 'Database not available'}), 500

        # Validate the database URL to prevent SSRF
        if not validate_db_url(db_url):
            return jsonify({'error': 'Invalid database URL'}), 500

        if request.method == 'GET':
            # Get value - construct safe URL
            safe_url = f"{db_url}/{key}"
            response = requests.get(safe_url, timeout=10)
            if response.status_code == 404:
                return jsonify({'error': 'Key not found'}), 404
            return jsonify({'key': key, 'value': response.text})

        elif request.method == 'POST':
            # Set value with input validation
            data = request.get_json()
            if not data or 'value' not in data:
                return jsonify({'error': 'Missing value in request'}), 400

            value = str(data.get('value', ''))
            # Limit value size to prevent abuse
            if len(value) > 1000000:  # 1MB limit
                return jsonify({'error': 'Value too large'}), 400

            safe_url = f"{db_url}/{key}"
            response = requests.post(safe_url, data=value, timeout=10)
            return jsonify({'success': True, 'key': key, 'value': value})

        elif request.method == 'DELETE':
            # Delete value
            safe_url = f"{db_url}/{key}"
            response = requests.delete(safe_url, timeout=10)
            if response.status_code == 404:
                return jsonify({'error': 'Key not found'}), 404
            return jsonify({'success': True, 'deleted': key})

    except requests.exceptions.Timeout:
        return jsonify({'error': 'Database request timeout'}), 500
    except Exception as e:
        return jsonify({'error': 'Database operation failed'}), 500

@app.route('/api/db/list')
def list_database_keys():
    """List all database keys with optional prefix"""
    try:
        import os
        import requests

        db_url = os.getenv("REPLIT_DB_URL")
        if not db_url:
            return jsonify({'error': 'Database not available'}), 500

        # Validate the database URL to prevent SSRF
        if not validate_db_url(db_url):
            return jsonify({'error': 'Invalid database URL'}), 500

        prefix = request.args.get('prefix', '')
        # Validate prefix to prevent injection
        if prefix and not validate_db_key(prefix):
            return jsonify({'error': 'Invalid prefix format'}), 400

        params = {'prefix': prefix} if prefix else {}

        response = requests.get(db_url, params=params, timeout=10)
        keys = [key.strip() for key in response.text.split('\n') if key.strip()]

        # Limit number of keys returned to prevent abuse
        if len(keys) > 1000:
            keys = keys[:1000]

        return jsonify({'keys': keys, 'count': len(keys)})

    except requests.exceptions.Timeout:
        return jsonify({'error': 'Database request timeout'}), 500
    except Exception as e:
        return jsonify({'error': 'Database operation failed'}), 500

@app.route('/test_database')
def test_database():
    """Test route to verify database integration"""
    try:
        # Test data
        test_job_data = {
            'job_id': 'test-' + str(uuid.uuid4())[:8],
            'tier': 'test',
            'full_name': 'Test User',
            'company_name': 'Test Company',
            'email': 'test@example.com',
            'status': 'test_entry',
            'price': 0
        }

        # Try logging to database
        success = save_to_database(test_job_data)

        return jsonify({
            'success': success,
            'message': 'Database integration test completed',
            'test_job_id': test_job_data['job_id']
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Database test failed: {str(e)}'
        })

@app.route('/process_realtime/<job_id>')
def process_realtime_analysis(job_id):
    try:
        # Validate job_id for security
        if not validate_job_id(job_id):
            return "Invalid job ID format", 400

        # Load job data
        file_path = safe_job_file_path(job_id)
        with open(file_path, 'r') as f:
            job_data = json.load(f)

        if job_data['tier'] not in ['free', 'professional']:
            return "Unauthorized", 403

        # Process the file and redirect to analyzer
        file_path = job_data['file_path']

        # Load the data into the analyzer
        global current_data
        try:
            df = pd.read_csv(file_path)
        except:
            try:
                df = pd.read_excel(file_path)
            except Exception as e:
                return f"Error reading file: {str(e)}", 500

        current_data = df

        # Add tier info to the redirect
        tier = job_data['tier']
        return redirect(url_for('analyzer', tier=tier, job_id=job_id))

    except Exception as e:
        return f"Error processing file: {str(e)}", 500

# Include all the existing analyzer routes (upload, cluster, etc.)
@app.route('/upload', methods=['POST'])
def upload_file():
    global current_data

    try:
        # Get tier from request
        tier = request.form.get('tier', 'free')
        max_size_mb = 100 if tier == 'free' else 100

        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                filename = secure_filename(file.filename)
                file_content = file.read()
                file_size = len(file_content)

                # Enforce size limits based on tier
                max_size_bytes = max_size_mb * 1024 * 1024
                if file_size > max_size_bytes:
                    return jsonify({
                        'success': False, 
                        'message': f'File too large ({file_size/(1024*1024):.1f}MB). {tier.capitalize()} tier supports files up to {max_size_mb}MB.'
                    })

                file_buffer = BytesIO(file_content)

                try:
                    if file_size > 50 * 1024 * 1024:
                        chunk_list = []
                        chunk_size = 10000
                        for chunk in pd.read_csv(file_buffer, encoding='utf-8', chunksize=chunk_size):
                            chunk_list.append(chunk)
                            if len(chunk_list) >= 10:
                                break
                        df = pd.concat(chunk_list, ignore_index=True)
                        file_type = "CSV (Large File - Sampled)"
                    else:
                        df = pd.read_csv(file_buffer, encoding='utf-8')
                        file_type = "CSV"
                except Exception as csv_error:
                    file_buffer.seek(0)
                    try:
                        df = pd.read_excel(file_buffer, engine="openpyxl")
                        file_type = "Excel"
                    except Exception as excel_error:
                        return jsonify({
                            'success': False, 
                            'message': f'Could not parse large file. CSV error: {str(csv_error)[:100]}... Excel error: {str(excel_error)[:100]}...'
                        })

                current_data = df

                def detect_column_type(series):
                    if series.dtype in ['object', 'string']:
                        avg_length = series.astype(str).str.len().mean()
                        unique_ratio = series.nunique() / len(series.dropna())
                        if avg_length > 10 or unique_ratio > 0.5:
                            return 'text'
                        else:
                            return 'categorical'
                    elif series.dtype in ['int64', 'float64', 'int32', 'float32']:
                        return 'numeric'
                    else:
                        return 'categorical'

                column_info = {}
                for col in df.columns:
                    col_type = detect_column_type(df[col])
                    column_info[col] = {
                        'type': col_type,
                        'unique_count': int(df[col].nunique()),
                        'null_count': int(df[col].isnull().sum())
                    }

                sample_data = {}
                for col in list(df.columns)[:5]:
                    samples = df[col].dropna().head(3).tolist()
                    sample_data[col] = [str(text)[:150] + ('...' if len(str(text)) > 150 else '') for text in samples]

                return jsonify({
                    'success': True,
                    'message': f'Loaded {file_type} file: {df.shape[0]:,} rows × {df.shape[1]} columns',
                    'columns': list(df.columns),
                    'all_columns': list(df.columns),
                    'column_info': column_info,
                    'preview': df.head().to_html(classes='table table-striped table-bordered'),
                    'sample_data': sample_data,
                    'ready_for_clustering': True
                })

        elif 'drive_link' in request.json:
            link = request.json['drive_link']

            try:
                fid = link.split("/d/")[1].split("/")[0]
            except:
                return jsonify({'success': False, 'message': 'Invalid Google Drive link format'})

            buf = download_from_drive(fid)
            content_preview = buf[:500]

            if (buf.lstrip().startswith(b"<!DOCTYPE html") or buf.lstrip().startswith(b"<html") or
                b'<title>Google Drive</title>' in content_preview):

                if b'Sorry, the file you have requested does not exist' in buf:
                    return jsonify({'success': False, 'message': 'File not found. Please check the Google Drive link is correct.'})
                elif b'You need permission' in buf or b'Access denied' in buf:
                    return jsonify({'success': False, 'message': 'Access denied. Please ensure the file is shared with "Anyone with the link can view" permissions.'})
                else:
                    return jsonify({'success': False, 'message': 'Received HTML page instead of file. The file may not be publicly accessible or the link format is incorrect.'})

            if len(buf) < 50:
                return jsonify({'success': False, 'message': f'File appears to be empty ({len(buf)} bytes). Please check sharing permissions and try again.'})

            # Enforce size limits based on tier
            tier = request.json.get('tier', 'free')
            max_size_mb = 100 if tier == 'free' else 100
            max_size_bytes = max_size_mb * 1024 * 1024
            file_size = len(buf)
            if file_size > max_size_bytes:
                return jsonify({
                    'success': False, 
                    'message': f'File too large ({file_size/(1024*1024):.1f}MB). {tier.capitalize()} tier supports files up to {max_size_mb}MB.'
                })

            file_buffer = BytesIO(buf)

            try:
                df = pd.read_csv(file_buffer, encoding='utf-8')
                file_type = "CSV"
            except Exception as csv_error:
                file_buffer.seek(0)
                try:
                    df = pd.read_excel(file_buffer, engine="openpyxl")
                    file_type = "Excel"
                except Exception as excel_error:
                    return jsonify({
                        'success': False, 
                        'message': f'Could not parse file. Please ensure it\'s a valid CSV or Excel file. CSV error: {str(csv_error)[:100]}... Excel error: {str(excel_error)[:100]}...'
                    })

            current_data = df

            def detect_column_type(series):
                if series.dtype in ['object', 'string']:
                    avg_length = series.astype(str).str.len().mean()
                    unique_ratio = series.nunique() / len(series.dropna())
                    if avg_length > 10 or unique_ratio > 0.5:
                        return 'text'
                    else:
                        return 'categorical'
                elif series.dtype in ['int64', 'float64', 'int32', 'float32']:
                    return 'numeric'
                else:
                    return 'categorical'

            column_info = {}
            for col in df.columns:
                col_type = detect_column_type(df[col])
                column_info[col] = {
                    'type': col_type,
                    'unique_count': int(df[col].nunique()),
                    'null_count': int(df[col].isnull().sum())
                }

            sample_data = {}
            for col in list(df.columns)[:5]:
                samples = df[col].dropna().head(3).tolist()
                sample_data[col] = [str(text)[:150] + ('...' if len(str(text)) > 150 else '') for text in samples]

            file_size_mb = len(buf) / (1024 * 1024)
            message = f'Loaded {file_type} from Google Drive: {df.shape[0]:,} rows × {df.shape[1]} columns'
            if file_size_mb > 10:
                message += f' (File size: {file_size_mb:.1f}MB)'

            return jsonify({
                'success': True,
                'message': message,
                'columns': list(df.columns),
                'all_columns': list(df.columns),
                'column_info': column_info,
                'preview': df.head().to_html(classes='table table-striped table-bordered'),
                'sample_data': sample_data,
                'ready_for_clustering': True,
                'file_size_mb': round(file_size_mb, 1),
                'is_large_file': file_size_mb > 50
            })

        return jsonify({'success': False, 'message': 'No file provided'})

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error processing file: {str(e)}'})

@app.route('/cluster', methods=['POST'])
def run_clustering():
    global current_data, clustering_result

    if current_data is None:
        return jsonify({'success': False, 'message': 'No data loaded'})

    try:
        data = request.json
        target_col = data['target_column']
        num_clusters = int(data['num_clusters'])
        max_features = int(data.get('max_features', 500))
        ngram_min = int(data.get('ngram_min', 1))
        ngram_max = int(data.get('ngram_max', 2))
        auto_optimize = data.get('auto_optimize', False)

        def detect_column_type(series):
            if series.dtype in ['object', 'string']:
                avg_length = series.astype(str).str.len().mean()
                unique_ratio = series.nunique() / len(series.dropna())
                if avg_length > 10 or unique_ratio > 0.5:
                    return 'text'
                else:
                    return 'categorical'
            elif series.dtype in ['int64', 'float64', 'int32', 'float32']:
                return 'numeric'
            else:
                return 'categorical'

        column_type = detect_column_type(current_data[target_col])

        column_info = {
            'name': target_col,
            'type': column_type,
            'data': current_data[target_col].tolist()
        }

        result = compute_advanced_clusters(
            current_data,
            column_info,
            num_clusters,
            max_features,
            (ngram_min, ngram_max),
            auto_optimize
        )

        clustering_result = result
        influencers, cluster_stats = get_enhanced_influencers(result)

        # Compute actual cluster counts and percentages
        counts = pd.Series(result["labels"]).value_counts().sort_index()
        counts = counts[counts.index != -1]  # Remove unclustered (-1)
        total = counts.sum()
        percentages = (counts / total * 100).round(1)
        
        # Build cluster labels with actual percentages
        cluster_labels = [f"Cluster {i} – {percentages[i]:.1f}%" for i in counts.index]
        cluster_values = counts.values
        
        # Create pie chart with actual distribution
        fig = px.pie(
            values=cluster_values,
            names=cluster_labels,
            title="Cluster Distribution"
        )
        
        # Update traces to show proper labels and percentages
        fig.update_traces(
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>'
        )
        
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            font=dict(size=11),
            height=500,
            width=700
        )
        distribution_chart = fig.to_json()

        silhouette_chart = None
        if result["silhouette_scores"]:
            fig = px.line(
                x=list(result["silhouette_scores"].keys()),
                y=list(result["silhouette_scores"].values()),
                title="Silhouette Score by Number of Clusters",
                labels={"x": "Number of Clusters", "y": "Silhouette Score"}
            )
            fig.add_vline(x=result["k"], line_dash="dash", line_color="red", 
                          annotation_text=f"Selected: {result['k']} clusters")
            silhouette_chart = fig.to_json()

        summary_data = []
        total_records = len(result["data"])

        for cid in range(result["k"]):
            summary_data.append({
                "Cluster": cid,
                "Count": cluster_stats[cid]["count"],
                "Percentage": f"{cluster_stats[cid]['count']/total_records*100:.1f}%",
                "Avg Text Length": f"{cluster_stats[cid]['avg_length']:.0f}",
                "Top Keywords": ", ".join(influencers[cid][:30])
            })

        response_data = {
            'success': True,
            'clusters_found': result["k"],
            'texts_clustered': len([l for l in result["labels"] if l != -1]),
            'total_records': result.get("total_records", len(result.get("data", []))),
            'distribution_chart': distribution_chart,
            'silhouette_chart': silhouette_chart,
            'summary_data': summary_data,
            'influencers': influencers,
            'cluster_stats': cluster_stats
        }

        if len(result.get("data", [])) > 10000:
            response_data['processing_info'] = {
                'is_large_dataset': True,
                'total_records': len(result.get("data", [])),
                'processing_time': f"Processing {len(result.get('data', [])):,} records"
            }

        available_columns = list(current_data.columns)
        response_data['available_columns'] = available_columns
        response_data['target_column'] = target_col
        response_data['data_type'] = column_type

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error during clustering: {str(e)}'})

@app.route('/generate_ai_prompt', methods=['POST'])
def generate_ai_prompt():
    global current_data, clustering_result

    if current_data is None or clustering_result is None:
        return jsonify({'success': False, 'message': 'No data or clustering results available'})

    try:
        data = request.json
        analysis_type = data.get('analysis_type', 'General Business Analyst')

        influencers, cluster_stats = get_enhanced_influencers(clustering_result, top_n=15)

        cluster_insights = []
        for cid in range(clustering_result['k']):
            cluster_insights.append({
                'id': cid,
                'count': cluster_stats[cid]["count"],
                'percentage': f"{cluster_stats[cid]['count']/len(clustering_result['data'])*100:.1f}%",
                'avg_length': cluster_stats[cid]['avg_length'],
                'keywords': influencers[cid][:20],
                'all_keywords': influencers[cid]
            })

        prompt = f"""You are SmartPrompt-AI, a Business-Analytics Expert specializing in {analysis_type}, I will share some snippet and responsibility but before you begin analysis ask me to upload Complete Analysis File.
Your mission: Analyze the attached comprehensive cluster analysis CSV file and deliver an end-to-end strategic whitepaper that combines deep data insights with actionable recommendations.

## Dataset Overview
- **Total Records**: {len(clustering_result['data']):,}
- **Clustered Records**: {len([l for l in clustering_result['labels'] if l != -1]):,}
- **Target Column**: {clustering_result.get('column_name', 'N/A')}
- **Data Type**: {clustering_result.get('data_type', 'N/A')}
- **Number of Clusters**: {clustering_result['k']}

## Attached Analysis File
**IMPORTANT**: A comprehensive cluster analysis CSV file is attached that contains:
- Complete cluster summary with detailed statistics
- All identified keywords and features for each cluster
- Cross-variable analysis (if performed)
- Overall dataset statistics
- Raw data with cluster assignments

Please analyze this attached CSV file thoroughly as it contains all the detailed data needed for your analysis.

## Cluster Summary Overview
"""

        for cluster in cluster_insights:
            cid = cluster['id']
            prompt += f"""
### Cluster {cid} - {cluster['count']} records ({cluster['percentage']})
**Key Characteristics**: {', '.join(cluster['keywords'][:8])}
**Extended Keywords**: {', '.join(cluster['keywords'][8:15])}
**Statistical Profile**: {cluster['count']} records, {cluster['percentage']} of total, avg length {cluster['avg_length']:.0f} chars
"""

        prompt += f"""

## Analysis Framework for {analysis_type}

### 1. **Data Mastery**
- Confirm understanding of the dataset schema and key fields
- Identify data quality issues and segmentation patterns
- Validate the clustering results align with business logic

### 2. **Executive Summary**
- Summarize the top 3 high-impact findings specific to {analysis_type} objectives
- Quantify the business impact potential
- State strategic implications for {analysis_type.lower()} operations

### 3. **Cluster-Driven Deep Dive**
For each of the {clustering_result['k']} clusters identified:

a. **Cluster Name**: Create a memorable, business-relevant label
b. **Core Insight**: One sentence capturing the defining characteristic
c. **Influencing Factors**: Analyze the provided keywords and features
d. **{analysis_type} Recommendations**: 2-3 prioritized actions with:
   - Implementation timeline
   - Resource requirements
   - Expected ROI/impact metrics
   - Risk mitigation strategies

### 4. **Cross-Cluster Synthesis**
- Identify patterns, overlaps, and outliers across clusters
- Highlight "opportunity zones" spanning multiple segments
- Map cluster relationships to business processes

### 5. **Strategic Roadmap**
- **30-Day Quick Wins**: Immediate actions with high impact
- **60-Day Strategic Initiatives**: Medium-term projects
- **90-Day Transformation Goals**: Long-term strategic changes
- Assign ownership and success metrics for each phase

### 6. **{analysis_type}-Specific KPIs & Metrics**
Define measurable outcomes aligned with {analysis_type.lower()} goals:
- Performance indicators
- Success benchmarks  
- Monitoring framework

## Deliverable Requirements
Return a comprehensive Markdown document titled:
**"Strategic Analysis Report: {analysis_type} Insights"**

Include:
- Executive dashboard with key metrics
- Detailed cluster analysis with actionable recommendations
- Implementation roadmap with timeline
- Risk assessment and mitigation strategies
- ROI projections where applicable

## Data-Driven Decision Framework
Base all recommendations on the cluster analysis provided, ensuring each insight is:
1. **Quantifiable**: Backed by cluster data and statistics
2. **Actionable**: Clear next steps for {analysis_type.lower()} teams
3. **Measurable**: Specific KPIs and success metrics
4. **Relevant**: Aligned with {analysis_type.lower()} strategic objectives

Begin your analysis now, using the clustering insights above as your foundation."""

        return jsonify({
            'success': True,
            'prompt': prompt,
            'analysis_type': analysis_type,
            'cluster_count': clustering_result['k'],
            'cluster_insights': cluster_insights
        })

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error generating AI prompt: {str(e)}'})

@app.route('/comprehensive_analysis', methods=['POST'])
def comprehensive_analysis():
    """Analyze cluster distribution across ALL data variables"""
    global current_data, clustering_result

    if current_data is None or clustering_result is None:
        return jsonify({'success': False, 'message': 'No data or clustering results available'})

    try:
        # Create analysis dataframe with cluster assignments
        analysis_df = current_data.copy().reset_index(drop=True)
        cluster_labels = clustering_result["labels"]
        
        if len(cluster_labels) != len(analysis_df):
            return jsonify({'success': False, 'message': f'Cluster labels length mismatch'})
        
        analysis_df['cluster_id'] = cluster_labels
        clustered_df = analysis_df[analysis_df['cluster_id'] != -1].copy()
        
        if len(clustered_df) == 0:
            return jsonify({'success': False, 'message': 'No clustered data found'})

        results = []
        target_column = clustering_result.get("column_name", "")
        
        # Analyze all columns except the target column
        for column in current_data.columns:
            if column == target_column:
                continue
                
            series = clustered_df[column].dropna()
            if len(series) == 0:
                continue
                
            is_numeric = pd.api.types.is_numeric_dtype(series)
            is_categorical = not is_numeric and series.nunique() <= 50
            
            if is_numeric:
                # Numeric variable analysis
                numeric_data = clustered_df[[column, 'cluster_id']].dropna()
                
                cluster_stats = []
                for cluster in sorted(numeric_data['cluster_id'].unique()):
                    cluster_data = numeric_data[numeric_data['cluster_id'] == cluster][column]
                    cluster_stats.append({
                        'Cluster': f'Cluster {cluster}',
                        'Count': len(cluster_data),
                        'Percentage': f"{len(cluster_data)/len(numeric_data)*100:.1f}%",
                        'Mean': round(cluster_data.mean(), 2),
                        'Median': round(cluster_data.median(), 2),
                        'Std Dev': round(cluster_data.std(), 2),
                        'Min': cluster_data.min(),
                        'Max': cluster_data.max()
                    })

                # Create distribution pie chart for numeric variable
                counts = numeric_data['cluster_id'].value_counts().sort_index()
                total = counts.sum()
                percentages = (counts / total * 100).round(1)
                
                # Build cluster labels with actual percentages
                cluster_labels = [f"Cluster {i} – {percentages[i]:.1f}%" for i in counts.index]
                cluster_values = counts.values
                
                fig = px.pie(
                    values=cluster_values,
                    names=cluster_labels,
                    title=f"Cluster Distribution for {column}"
                )
                fig.update_traces(
                    textinfo='label+percent',
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>'
                )
                
                results.append({
                    'variable': column,
                    'type': 'numeric',
                    'chart': fig.to_json(),
                    'summary_table': pd.DataFrame(cluster_stats).to_html(classes='table table-striped', index=False)
                })
                
            elif is_categorical:
                # Categorical variable analysis
                categorical_data = clustered_df[[column, 'cluster_id']].dropna()
                
                # Create crosstab
                crosstab = pd.crosstab(categorical_data['cluster_id'], categorical_data[column])
                
                # Calculate cluster distribution for this variable
                counts = categorical_data['cluster_id'].value_counts().sort_index()
                total = counts.sum()
                percentages = (counts / total * 100).round(1)
                
                # Build cluster labels with actual percentages
                cluster_labels = [f"Cluster {i} – {percentages[i]:.1f}%" for i in counts.index]
                cluster_values = counts.values
                
                fig = px.pie(
                    values=cluster_values,
                    names=cluster_labels,
                    title=f"Cluster Distribution for {column}"
                )
                fig.update_traces(
                    textinfo='label+percent',
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>'
                )
                
                # Create summary statistics
                cluster_stats = []
                for cluster in sorted(categorical_data['cluster_id'].unique()):
                    cluster_data = categorical_data[categorical_data['cluster_id'] == cluster]
                    top_categories = cluster_data[column].value_counts().head(3)
                    
                    cluster_stats.append({
                        'Cluster': f'Cluster {cluster}',
                        'Count': len(cluster_data),
                        'Percentage': f"{len(cluster_data)/len(categorical_data)*100:.1f}%",
                        'Top Categories': ', '.join([f"{cat} ({count})" for cat, count in top_categories.items()])
                    })
                
                results.append({
                    'variable': column,
                    'type': 'categorical',
                    'chart': fig.to_json(),
                    'summary_table': pd.DataFrame(cluster_stats).to_html(classes='table table-striped', index=False)
                })

        return jsonify({
            'success': True,
            'results': results,
            'total_variables_analyzed': len(results),
            'message': f'Comprehensive analysis completed for {len(results)} variables'
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'message': f'Error in comprehensive analysis: {str(e)}', 'traceback': traceback.format_exc()})

@app.route('/cross_analysis', methods=['POST'])
def cross_analysis():
    global current_data, clustering_result

    if current_data is None or clustering_result is None:
        return jsonify({'success': False, 'message': 'No data or clustering results available'})

    try:
        data = request.json
        variables = data['variables']

        if not isinstance(variables, list):
            variables = [variables]

        for variable in variables:
            if variable not in current_data.columns:
                return jsonify({'success': False, 'message': f'Variable "{variable}" not found in dataset'})

        # Create df_map (original DataFrame plus cluster column)
        df_map = current_data.copy().reset_index(drop=True)
        cluster_labels = clustering_result["labels"]
        
        if len(cluster_labels) != len(df_map):
            return jsonify({'success': False, 'message': f'Cluster labels length ({len(cluster_labels)}) does not match data length ({len(df_map)})'})
        
        df_map['cluster'] = cluster_labels
        # Remove unclustered data (cluster = -1)
        df_map = df_map[df_map['cluster'] != -1].copy()
        
        # Ensure cluster column exists and has valid data
        if 'cluster' not in df_map.columns or df_map['cluster'].isna().all():
            return jsonify({'success': False, 'message': 'No valid cluster assignments found'})
        
        if len(df_map) == 0:
            return jsonify({'success': False, 'message': 'No clustered data found for analysis'})

        results = []
        
        # Overall Cluster Distribution (always first)
        counts = df_map['cluster'].value_counts().sort_index()
        total = counts.sum()
        percentages = (counts / total * 100).round(1)
        
        # Build cluster labels with actual percentages
        cluster_labels = [f"Cluster {i} – {percentages[i]:.1f}%" for i in counts.index]
        cluster_values = counts.values
        
        overall_fig = px.pie(
            values=cluster_values,
            names=cluster_labels,
            title="Overall Cluster Distribution"
        )
        overall_fig.update_traces(
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>'
        )
        
        results.append({
            'variable': 'Overall Distribution',
            'chart': overall_fig.to_json(),
            'summary_table': f'<div class="alert alert-info"><strong>Total Records:</strong> {total:,}<br><strong>Clusters Found:</strong> {len(counts)}</div>',
            'variable_type': 'overall',
            'correlation_analysis': False
        })

        # Per-Variable Breakdown
        for variable in variables:
            series = df_map[variable].dropna()
            if len(series) == 0:
                continue
                
            is_numeric = pd.api.types.is_numeric_dtype(series)
            is_date = False
            
            # Check for date columns
            if not is_numeric:
                is_date = pd.api.types.is_datetime64_any_dtype(series)
                if not is_date:
                    try:
                        pd.to_datetime(series.head(10), errors="raise")
                        is_date = True
                    except:
                        is_date = False

            if is_numeric:
                # Numeric columns: Bar chart of mean & median values by cluster
                clean_data = df_map[[variable, 'cluster']].dropna()
                
                # Compute actual cluster counts
                counts = clean_data['cluster'].value_counts().sort_index()
                
                # Aggregate mean and median by cluster
                stats = (
                    clean_data
                    .groupby('cluster')[variable]
                    .agg(['mean', 'median'])
                    .reindex(counts.index)  # preserves cluster order
                )
                
                # Build x-axis labels with record counts
                x_labels = [f"Cluster {i} ({counts[i]} recs)" for i in counts.index]
                
                # Create bar chart with mean and median
                fig = px.bar(
                    stats,
                    x=x_labels,
                    y=['mean', 'median'],
                    barmode='group',
                    title=f"Mean & Median by Cluster: {variable}"
                )
                fig.update_layout(
                    xaxis_title="Cluster (with record counts)",
                    yaxis_title=variable,
                    height=400
                )
                
                # Summary table with actual cluster statistics
                summary_data = []
                for cluster_id in sorted(clean_data['cluster'].unique()):
                    cluster_data = clean_data[clean_data['cluster'] == cluster_id][variable]
                    cluster_count = len(cluster_data)
                    total_count = len(clean_data)
                    
                    if cluster_count > 0:
                        summary_data.append({
                            'Cluster': f"Cluster {cluster_id}",
                            'Count': cluster_count,
                            'Percentage': f"{cluster_count/total_count*100:.1f}%",
                            'Mean': f"{cluster_data.mean():.2f}",
                            'Median': f"{cluster_data.median():.2f}",
                            'Std Dev': f"{cluster_data.std():.2f}",
                            'Min': f"{cluster_data.min():.2f}",
                            'Max': f"{cluster_data.max():.2f}"
                        })
                
                summary_table = pd.DataFrame(summary_data).to_html(classes='table table-striped table-bordered', index=False)
                
            elif is_date:
                # Date/time columns: Line chart of row counts over time per cluster
                date_data = df_map[[variable, 'cluster']].copy()
                try:
                    # Compute actual cluster counts
                    counts = date_data['cluster'].value_counts().sort_index()
                    
                    # Process dates and create time series pivot
                    ser_dt = pd.to_datetime(date_data[variable], errors="coerce")
                    time_pivot = (
                        date_data.assign(_d=ser_dt.dt.date)
                                .dropna(subset=['_d'])
                                .groupby(['_d', 'cluster'])
                                .size()
                                .unstack(fill_value=0)
                                .reindex(columns=counts.index, fill_value=0)
                    )
                    
                    # Create line chart with actual cluster counts in legend
                    fig = px.line(
                        time_pivot,
                        x=time_pivot.index,
                        y=[f"Cluster {i} ({counts[i]})" for i in counts.index],
                        title=f"Time Series by Cluster: {variable}"
                    )
                    fig.update_layout(
                        xaxis_title=variable,
                        yaxis_title="Count",
                        height=400
                    )
                    
                    # Summary table with actual date range info and cluster counts
                    summary_data = []
                    for cluster in tmp.columns:
                        cluster_dates = date_data[date_data['cluster'] == cluster]
                        total_count = tmp[cluster].sum()
                        total_overall = len(date_data)
                        peak_date = tmp[cluster].idxmax() if tmp[cluster].sum() > 0 else None
                        peak_count = tmp[cluster].max() if tmp[cluster].sum() > 0 else 0
                        
                        # Get actual date range for this cluster
                        cluster_date_values = cluster_dates[variable].dropna()
                        if len(cluster_date_values) > 0:
                            date_range = f"{pd.to_datetime(cluster_date_values).min().date()} to {pd.to_datetime(cluster_date_values).max().date()}"
                        else:
                            date_range = "No data"
                        
                        summary_data.append({
                            'Cluster': f"Cluster {cluster}",
                            'Total Records': int(total_count),
                            'Percentage': f"{total_count/total_overall*100:.1f}%",
                            'Date Range': date_range,
                            'Peak Date': str(peak_date) if peak_date else "N/A",
                            'Peak Count': int(peak_count)
                        })
                    
                    summary_table = pd.DataFrame(summary_data).to_html(classes='table table-striped table-bordered', index=False)
                    
                except Exception as e:
                    # Fallback for problematic date parsing
                    fig = go.Figure()
                    fig.add_annotation(text=f"Date parsing failed: {str(e)}", x=0.5, y=0.5, showarrow=False)
                    summary_table = f"<div class='alert alert-warning'>Date analysis failed for {variable}</div>"
                
            else:
                # Categorical columns: Bar chart of counts per category by cluster
                cat_data = df_map[[variable, 'cluster']].dropna()
                
                # Get top categories to avoid overcrowding
                top_categories = cat_data[variable].value_counts().head(10).index
                cat_data_filtered = cat_data[cat_data[variable].isin(top_categories)]
                
                # Compute actual cluster counts
                counts = cat_data_filtered['cluster'].value_counts().sort_index()
                
                # Create pivot table for category counts by cluster
                pivot = (
                    cat_data_filtered
                    .groupby(['cluster', variable])
                    .size()
                    .unstack(fill_value=0)
                    .reindex(counts.index)  # preserves cluster order
                )
                
                # Build x-axis labels with record counts
                x_labels = [f"Cluster {i} ({counts[i]} recs)" for i in counts.index]
                
                # Create grouped bar chart
                fig = px.bar(
                    pivot,
                    x=x_labels,
                    y=pivot.columns,
                    title=f"Category Counts by Cluster: {variable}",
                    labels={"value": "Count"}
                )
                fig.update_layout(
                    xaxis_title="Cluster (with record counts)",
                    height=400
                )
                
                # Summary table with actual category percentages per cluster
                summary_data = []
                for cluster in sorted(cat_data['cluster'].unique()):
                    cluster_data = cat_data[cat_data['cluster'] == cluster]
                    top_cats = cluster_data[variable].value_counts().head(3)
                    total_in_cluster = len(cluster_data)
                    total_overall = len(cat_data)
                    
                    summary_data.append({
                        'Cluster': f"Cluster {cluster}",
                        'Total Records': total_in_cluster,
                        'Percentage': f"{total_in_cluster/total_overall*100:.1f}%",
                        'Top Category': top_cats.index[0] if len(top_cats) > 0 else "None",
                        'Top Count (%)': f"{top_cats.iloc[0]} ({top_cats.iloc[0]/total_in_cluster*100:.1f}%)" if len(top_cats) > 0 else "0",
                        'Categories': len(cluster_data[variable].unique())
                    })
                
                summary_table = pd.DataFrame(summary_data).to_html(classes='table table-striped table-bordered', index=False)

            results.append({
                'variable': variable,
                'chart': fig.to_json(),
                'summary_table': summary_table,
                'variable_type': 'date' if is_date else 'numeric' if is_numeric else 'categorical',
                'correlation_analysis': True
            })

        return jsonify({
            'success': True,
            'results': results,
            'message': f'Smart visual analysis completed for {len(variables)} variable(s) with overall cluster distribution'
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'message': f'Error in cross-analysis: {str(e)}', 'traceback': traceback.format_exc()})

@app.route('/export/<export_type>')
def export_data(export_type):
    global current_data, clustering_result

    if current_data is None or clustering_result is None:
        return jsonify({'success': False, 'message': 'No data or clustering results available'})

    try:
        if export_type == 'summary':
            output = BytesIO()
            influencers, cluster_stats = get_enhanced_influencers(clustering_result)
            summary_data = []
            total_records = len(clustering_result["data"])

            for cid in range(clustering_result["k"]):
                summary_data.append({
                    "Cluster": cid,
                    "Count": cluster_stats[cid]["count"],
                    "Percentage": f"{cluster_stats[cid]['count']/total_records*100:.1f}%",
                    "Avg Text Length": f"{cluster_stats[cid]['avg_length']:.0f}",
                    "Top Keywords": ", ".join(influencers[cid][:30])
                })

            summary_df = pd.DataFrame(summary_data)

            csv_content = "=== CLUSTER SUMMARY ===\n"
            csv_content += summary_df.to_csv(index=False)
            csv_content += "\n\n"

            # Add cross-variable analysis for all numeric and categorical columns
            analysis_df = current_data.copy()
            analysis_df['cluster_id'] = clustering_result["labels"]
            analysis_df = analysis_df[analysis_df['cluster_id'] != -1]

            target_column = clustering_result.get("column_name", "")

            # Analyze numeric columns
            numeric_columns = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
            if 'cluster_id' in numeric_columns:
                numeric_columns.remove('cluster_id')

            for col in numeric_columns:
                if col != target_column and analysis_df[col].notna().sum() > 0:
                    csv_content += f"=== {col.upper()} ANALYSIS ===\n"

                    # Calculate statistics for each cluster
                    cluster_analysis = []
                    for cluster_id in sorted(analysis_df['cluster_id'].unique()):
                        cluster_data = analysis_df[analysis_df['cluster_id'] == cluster_id][col].dropna()
                        if len(cluster_data) > 0:
                            cluster_analysis.append({
                                'cluster_id': cluster_id,
                                'Count': len(cluster_data),
                                'Mean': round(cluster_data.mean(), 2),
                                'Median': round(cluster_data.median(), 2),
                                'Std Dev': round(cluster_data.std(), 2),
                                'Min': cluster_data.min(),
                                'Max': cluster_data.max(),
                                '% of Total': round(len(cluster_data) / len(analysis_df) * 100, 1)
                            })

                    if cluster_analysis:
                        cluster_df = pd.DataFrame(cluster_analysis)
                        csv_content += cluster_df.to_csv(index=False)
                        csv_content += "\n"

            # Analyze categorical columns
            categorical_columns = analysis_df.select_dtypes(include=['object', 'string']).columns.tolist()
            if target_column in categorical_columns:
                categorical_columns.remove(target_column)

            for col in categorical_columns:
                if col != target_column and analysis_df[col].notna().sum() > 0:
                    # Only analyze if column has reasonable number of unique values
                    unique_count = analysis_df[col].nunique()
                if unique_count <= 50:  # Only analyze if reasonable number of categories
                    csv_content += f"=== {col.upper()} ANALYSIS ===\n"
                    cross_tab = pd.crosstab(analysis_df['cluster_id'], analysis_df[col])
                    csv_content += cross_tab.to_csv()
                    csv_content += "\n"

            output.write(csv_content.encode('utf-8'))
            output.seek(0)

            return send_file(
                output,
                mimetype='text/csv',
                as_attachment=True,
                download_name='cluster_summary.csv'
            )
        elif export_type == 'full':
            output = BytesIO()
            df = current_data.copy()
            df['cluster'] = clustering_result['labels']
            df.to_csv(output, index=False, encoding='utf-8')
            output.seek(0)

            return send_file(
                output,
                mimetype='text/csv',
                as_attachment=True,
                download_name='clustered_data.csv'
            )
        else:
            return jsonify({'success': False, 'message': 'Invalid export type'})

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error during export: {str(e)}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)