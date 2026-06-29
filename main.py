import logging 
from fastapi import FASTAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime


load_dotenv()

inngest_client=inngest.Inngest(
    app_id="rag_app",
    logging=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)

app=FASTAPI()
app.get()

    
inngest.fast_api.serve(app,inngest_client,[])


