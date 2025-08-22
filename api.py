from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from claim_processor import ClaimProcessor
import tempfile
import os
from typing import List

app = FastAPI(title="Claim Resubmission API", description="API for processing EMR data and identifying resubmission candidates")

@app.post("/process-claims/", response_model=List[dict])
async def process_claims(alpha_file: UploadFile = File(...), beta_file: UploadFile = File(...)):
    """
    Process claim files from two EMR sources and return eligible claims for resubmission.
    """
    # Validate file types
    if alpha_file.content_type not in ['text/csv', 'application/vnd.ms-excel']:
        raise HTTPException(400, "Alpha file must be a CSV")
    if beta_file.content_type != 'application/json':
        raise HTTPException(400, "Beta file must be a JSON")

    alpha_path, beta_path = None, None
    
    try:
        # Save uploaded files to temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as alpha_temp:
            content = await alpha_file.read()
            alpha_temp.write(content)
            alpha_path = alpha_temp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as beta_temp:
            content = await beta_file.read()
            beta_temp.write(content)
            beta_path = beta_temp.name

        # Process the files
        processor = ClaimProcessor()
        processor.process_files(alpha_path, beta_path)
        
        # Include metrics in the response for better debugging
        metrics = processor.get_metrics()
        response = {
            "candidates": processor.resubmission_candidates,
            "metrics": metrics
        }
        
        return response

    except Exception as e:
        raise HTTPException(500, f"Error processing files: {str(e)}")
    
    finally:
        # Clean up temporary files in finally block to ensure it happens
        for path in [alpha_path, beta_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass  # Don't let cleanup errors mask the original errors


                
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "claim-resubmission-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)