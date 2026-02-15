from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from api.auth import get_admin_user
from execution.rag.ingest_pdf import PDFIngestor
import shutil
import os

router = APIRouter(prefix="/admin", tags=["admin"])

@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    procedure_type: str = "general",
    language: str = "en",
    user: dict = Depends(get_admin_user)
):
    """Admin endpoint to upload and ingest a PDF."""
    try:
        # Save temp file
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Ingest
        ingestor = PDFIngestor()
        ingestor.ingest(temp_path, metadata={"procedure_type": procedure_type, "language": language, "filename": file.filename})
        
        # Cleanup
        os.remove(temp_path)
        
        return {"status": "success", "message": f"Ingested {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
