from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from supabase import create_client, Client
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
import os
import sqlite3
import smtplib
import uuid
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import secrets
import hashlib

load_dotenv()

# Initialize Supabase client
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_API_KEY")
service_role_key: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(url, service_role_key)

app = FastAPI(title="Delete Server", description="User deletion service with email and user ID methods")

# Email configuration (add these to your .env file)
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# Pydantic models
class EmailDeleteRequest(BaseModel):
    email: EmailStr

class UserIdDeleteRequest(BaseModel):
    user_id: str

class DeleteConfirmation(BaseModel):
    token: str

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('delete_requests.db')
    cursor = conn.cursor()
    
    # Table for email-based deletion requests
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS email_delete_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            is_deleted BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table for user ID deletion tokens
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_delete_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            email TEXT NOT NULL,
            token TEXT NOT NULL UNIQUE,
            expires_at TIMESTAMP NOT NULL,
            is_used BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

def get_db_connection():
    return sqlite3.connect('delete_requests.db')

def send_confirmation_email(email: str, token: str):
    """Send confirmation email with deletion link using Gmail SMTP"""
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        raise HTTPException(status_code=500, detail="Gmail configuration not set. Please set EMAIL_ADDRESS and EMAIL_PASSWORD in .env")
    
    subject = "Confirm Account Deletion"
    confirmation_link = f"{BACKEND_URL}/confirm-delete?token={token}"
    
    body = f"""
    <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background-color: #f8f9fa; padding: 30px; border-radius: 10px;">
                <h2 style="color: #dc3545; text-align: center;">Account Deletion Confirmation</h2>
                <p style="font-size: 16px; line-height: 1.6;">You have requested to delete your account. If this was you, please click the button below to confirm the deletion:</p>
                
                <div style="text-align: center; margin: 30px 0;">
                    <a href="{confirmation_link}" 
                       style="background-color: #dc3545; 
                              color: white; 
                              padding: 15px 30px; 
                              text-decoration: none; 
                              border-radius: 5px; 
                              font-weight: bold;
                              display: inline-block;">
                        Confirm Account Deletion
                    </a>
                </div>
                
                <div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107;">
                    <p style="margin: 0; color: #856404;"><strong>Important:</strong></p>
                    <ul style="color: #856404; margin: 10px 0;">
                        <li>This link will expire in 24 hours</li>
                        <li>Once deleted, your account cannot be recovered</li>
                        <li>If you did not request this deletion, please ignore this email</li>
                    </ul>
                </div>
                
                <p style="font-size: 14px; color: #6c757d; margin-top: 20px;">
                    If the button doesn't work, you can copy and paste this link into your browser:<br>
                    <a href="{confirmation_link}" style="color: #007bff; word-break: break-all;">{confirmation_link}</a>
                </p>
            </div>
        </body>
    </html>
    """
    
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = email
    
    html_part = MIMEText(body, 'html')
    msg.attach(html_part)
    
    try:
        print(f"Attempting to send email to {email}")
        print(f"Using SMTP server: {SMTP_SERVER}:{SMTP_PORT}")
        print(f"From email: {EMAIL_ADDRESS}")
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            print("STARTTLS successful")
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            print("Login successful")
            server.send_message(msg)
            print(f"Confirmation email sent successfully to {email}")
            
    except smtplib.SMTPAuthenticationError as e:
        print(f"SMTP Authentication Error: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Gmail authentication failed. Please check: 1) 2FA is enabled on Gmail, 2) Using App Password (not regular password), 3) EMAIL_ADDRESS and EMAIL_PASSWORD are correct in .env"
        )
    except smtplib.SMTPRecipientsRefused as e:
        print(f"SMTP Recipients Refused: {e}")
        raise HTTPException(status_code=400, detail="Invalid recipient email address")
    except smtplib.SMTPServerDisconnected as e:
        print(f"SMTP Server Disconnected: {e}")
        raise HTTPException(status_code=500, detail="SMTP server connection failed")
    except Exception as e:
        print(f"Email sending failed with error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Delete Server API", "version": "1.0"}

# Method 1: Email-based deletion request
@app.post("/request-delete-by-email")
async def request_delete_by_email(request: EmailDeleteRequest):
    """Submit a deletion request with email address"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if email already has a pending request
        cursor.execute(
            "SELECT id FROM email_delete_requests WHERE email = ? AND is_deleted = FALSE",
            (request.email,)
        )
        existing = cursor.fetchone()
        
        if existing:
            conn.close()
            return {"message": "Deletion request already exists for this email", "status": "pending"}
        
        # Insert new deletion request
        cursor.execute(
            "INSERT INTO email_delete_requests (email) VALUES (?)",
            (request.email,)
        )
        
        conn.commit()
        conn.close()
        
        return {
            "message": "Deletion request submitted successfully",
            "email": request.email,
            "status": "pending"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# Method 2: User ID-based deletion with email confirmation
@app.post("/request-delete-by-user-id")
async def request_delete_by_user_id(request: UserIdDeleteRequest):
    """Request deletion by user ID - sends confirmation email"""
    try:
        # First, get user email from Supabase
        user_response = supabase.auth.admin.get_user_by_id(request.user_id)
        
        if not user_response.user:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_email = user_response.user.email
        print(f"User email for ID {request.user_id}: {user_email}")
        
        if not user_email:
            raise HTTPException(status_code=400, detail="User email not found")
        
        # Generate secure token
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=24)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check for existing unused token
        cursor.execute(
            "SELECT id FROM user_delete_tokens WHERE user_id = ? AND is_used = FALSE AND expires_at > ?",
            (request.user_id, datetime.now())
        )
        existing = cursor.fetchone()
        
        if existing:
            conn.close()
            return {"message": "Deletion confirmation email already sent", "status": "pending"}
        
        # Store token
        cursor.execute(
            "INSERT INTO user_delete_tokens (user_id, email, token, expires_at) VALUES (?, ?, ?, ?)",
            (request.user_id, user_email, token, expires_at)
        )
        
        conn.commit()
        conn.close()
        
        # Send confirmation email synchronously to handle errors properly
        try:
            send_confirmation_email(user_email, token)
            return {
                "message": "Confirmation email sent successfully",
                "user_id": request.user_id,
                "email": user_email,
                "status": "confirmation_sent"
            }
        except HTTPException as email_error:
            # If email fails, we should still return success since token is stored
            # but inform about email issue
            return {
                "message": "Deletion request created but email failed to send",
                "user_id": request.user_id,
                "email": user_email,
                "status": "email_failed",
                "error": str(email_error.detail),
                "note": "User can manually visit the deletion link or contact support"
            }
        
    except Exception as e:
        if "User not found" in str(e):
            raise HTTPException(status_code=404, detail="User not found")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/confirm-delete")
async def confirm_delete(token: str):
    """Confirm deletion via email link"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Find and validate token
        cursor.execute(
            "SELECT user_id, email, expires_at, is_used FROM user_delete_tokens WHERE token = ?",
            (token,)
        )
        token_data = cursor.fetchone()
        
        if not token_data:
            conn.close()
            return HTMLResponse(
                content="<h1>Invalid Token</h1><p>The deletion link is invalid or has expired.</p>",
                status_code=400
            )
        
        user_id, email, expires_at, is_used = token_data
        
        # Check if token is expired
        if datetime.fromisoformat(expires_at) < datetime.now():
            conn.close()
            return HTMLResponse(
                content="<h1>Expired Link</h1><p>The deletion link has expired. Please request a new one.</p>",
                status_code=400
            )
        
        # Check if token is already used
        if is_used:
            conn.close()
            return HTMLResponse(
                content="<h1>Already Used</h1><p>This deletion link has already been used.</p>",
                status_code=400
            )
        
        # Mark token as used
        cursor.execute(
            "UPDATE user_delete_tokens SET is_used = TRUE WHERE token = ?",
            (token,)
        )
        
        conn.commit()
        conn.close()
        
        # Delete user from Supabase
        try:
            supabase.auth.admin.delete_user(user_id)
            
            return HTMLResponse(
                content="""
                <html>
                    <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px;">
                        <h1 style="color: #28a745;">Account Deleted Successfully</h1>
                        <p>Your account has been permanently deleted from our system.</p>
                        <p>Thank you for using our service.</p>
                    </body>
                </html>
                """
            )
            
        except Exception as e:
            return HTMLResponse(
                content=f"<h1>Error</h1><p>Failed to delete account: {str(e)}</p>",
                status_code=500
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error confirming deletion: {str(e)}")

# Admin endpoints for managing deletion requests
@app.get("/admin/email-delete-requests")
async def get_email_delete_requests():
    """Get all email deletion requests (admin only)"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, email, is_deleted, created_at FROM email_delete_requests ORDER BY created_at DESC"
        )
        
        requests = []
        for row in cursor.fetchall():
            requests.append({
                "id": row[0],
                "email": row[1],
                "is_deleted": row[2],
                "created_at": row[3]
            })
        
        conn.close()
        return {"requests": requests}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/admin/process-email-deletion/{request_id}")
async def process_email_deletion(request_id: int):
    """Process an email deletion request (admin only)"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the request
        cursor.execute(
            "SELECT email, is_deleted FROM email_delete_requests WHERE id = ?",
            (request_id,)
        )
        request_data = cursor.fetchone()
        
        if not request_data:
            conn.close()
            raise HTTPException(status_code=404, detail="Request not found")
        
        email, is_deleted = request_data
        
        if is_deleted:
            conn.close()
            return {"message": "Request already processed", "status": "already_deleted"}
        
        # Delete the request
        cursor.execute(
            "DELETE FROM email_delete_requests WHERE id = ?",
            (request_id,)
        )
        
        conn.commit()
        conn.close()
        
        # Here you would implement the actual deletion logic
        # For example, delete from your main user database
        
        return {
            "message": "Email deletion request processed",
            "request_id": request_id,
            "email": email,
            "status": "processed"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing deletion: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)