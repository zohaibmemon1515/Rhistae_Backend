from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, function_tool
from dotenv import load_dotenv
import os
import requests
import urllib.parse
import uvicorn

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SanityConnection:
    def __init__(self):
        self.project_id = os.getenv("SANITY_PROJECT_ID")
        self.dataset = os.getenv("SANITY_DATASET")
        self.api_version = os.getenv("SANITY_API_VERSION", "2023-08-01")
        self.token = os.getenv("SANITY_API_TOKEN")
        
    def validate_connection(self):
        if not all([self.project_id, self.dataset, self.token]):
            return False, "Missing Sanity configuration"
        
        test_url = f"https://{self.project_id}.api.sanity.io/v{self.api_version}/data/query/{self.dataset}?query=count(*)"
        headers = {"Authorization": f"Bearer {self.token}"}
        
        try:
            response = requests.get(test_url, headers=headers, timeout=5)
            if response.status_code == 401:
                return False, "Invalid Sanity API token"
            return True, "Connection successful"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

@function_tool()
def send_whatsapp_message(number: str, message: str):
    """Send WhatsApp message using UltraMSG API"""
    instance_id = os.getenv("ULTRASMART_INSTANCE_ID")
    api_key = os.getenv("ULTRASMART_API_KEY")
    
    if not all([instance_id, api_key]):
        return "❌ WhatsApp API not configured"
    
    url = f"https://api.ultramsg.com/{instance_id}/messages/chat"
    payload = {
        "token": api_key,
        "to": number,
        "body": message
    }
    
    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            return "✅ WhatsApp message sent successfully!"
        return f"❌ WhatsApp API error: {response.text}"
    except Exception as e:
        return f"❌ WhatsApp connection failed: {str(e)}"

@function_tool()
def find_match(age: int, gender: str):
    sanity = SanityConnection()
    valid, msg = sanity.validate_connection()
    if not valid:
        return []
    
    query = f"""
    *[_type == "profile" &&
      gender != "{gender}" && 
      age >= {max(18, int(age)-5)} && 
      age <= {int(age)+5}
    ]{{
      name,
      age,
      gender,
      city,
      education,
      profession
    }}[0...5]
    """
    
    try:
        encoded_query = urllib.parse.quote(query.strip())
        url = f"https://{sanity.project_id}.api.sanity.io/v{sanity.api_version}/data/query/{sanity.dataset}?query={encoded_query}"
        headers = {"Authorization": f"Bearer {sanity.token}"}
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        return data.get("result", [])
    except Exception:
        return []


GEMINI_API = os.getenv("GEMINI_Api")
if not GEMINI_API:
    raise ValueError("Missing GEMINI_Api in environment variables")

external_client = AsyncOpenAI(
    api_key=GEMINI_API,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

matchmaker_agent = Agent(
    name="Rhistaé – Matchmaker",
    instructions="""
You are Rhistaé, a matchmaker agent that finds suitable matches based on user preferences.
Your task is to:
1. First find matches using the find_match tool with the provided age and gender.
2. If matches found:
   - Format each match with:
     💘 Match Found 💘
     Here are the details of your matches:
    ---------------
     Name: {name}
     Age: {age}
     Gender: {gender}
     City: {city}
     Education: {education}
     Profession: {profession}

     if you like any of these matches, please type 'Interested' to proceed.
   - Send via WhatsApp using send_whatsapp_message
   - Return "Matches sent to your WhatsApp!"
3. If no matches:
   - Return "No suitable matches found"
""",
    tools=[find_match, send_whatsapp_message]
)

class MatchRequest(BaseModel):
    age: int
    gender: str
    number: str

@app.post("/api/find-match/")
async def get_match(data: MatchRequest):
    try:
        prompt = f"""
Find matches for:
- Age: {data.age}
- Gender: {data.gender}

Then send to WhatsApp: {data.number}
"""
        result = await Runner.run(
            starting_agent=matchmaker_agent,
            input=prompt,
            run_config=config,
        )
        
        return {"message": result.final_output, "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
