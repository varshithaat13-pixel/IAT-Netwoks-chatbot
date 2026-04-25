import json
import uuid

def generate_chunks():
    with open('IAT_Networks_Structured.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunks = []
    source = "company_document_json"

    # 1. Company Overview
    chunks.append({
        "id": "chunk_001",
        "text": data["company_overview"],
        "section": "company_overview",
        "sub_section": "overview",
        "intent": "company_info",
        "keywords": ["IAT Networks", "Vellore consulting", "outsourcing firm", "IT services", "BPO"],
        "priority": "high",
        "source": source
    })

    # 2. About - Mission & Vision
    chunks.append({
        "id": "chunk_002",
        "text": f"Mission: {data['about_company']['mission']}\nVision: {data['about_company']['vision']}",
        "section": "about_company",
        "sub_section": "mission_vision",
        "intent": "company_info",
        "keywords": ["mission statement", "vision statement", "smart technology", "global partner"],
        "priority": "medium",
        "source": source
    })

    # 3. About - Foundation
    chunks.append({
        "id": "chunk_003",
        "text": f"IAT Networks was founded in {data['about_company']['founded']} by {data['about_company']['founder']}.",
        "section": "about_company",
        "sub_section": "history",
        "intent": "company_info",
        "keywords": ["founded date", "founder name", "Ishwarya S", "history"],
        "priority": "medium",
        "source": source
    })

    # 4. Contact - Phone (Direct Answer)
    chunks.append({
        "id": "chunk_004",
        "text": f"The contact phone number for IAT Networks is {data['contact_information']['phone']}.",
        "section": "contact_information",
        "sub_section": "phone_number",
        "intent": "contact_info",
        "keywords": ["phone number", "call", "contact phone", "support number"],
        "priority": "high",
        "source": source
    })

    # 5. Contact - Email (Direct Answer)
    chunks.append({
        "id": "chunk_005",
        "text": f"You can reach IAT Networks via email at {data['contact_information']['email']}.",
        "section": "contact_information",
        "sub_section": "email_address",
        "intent": "contact_info",
        "keywords": ["email address", "contact email", "hr email", "write to us"],
        "priority": "high",
        "source": source
    })

    # 6. Contact - Address (Direct Answer)
    chunks.append({
        "id": "chunk_006",
        "text": f"IAT Networks is located at {data['contact_information']['address']}.",
        "section": "contact_information",
        "sub_section": "office_location",
        "intent": "contact_info",
        "keywords": ["office address", "location", "headquarters", "Vellore office"],
        "priority": "high",
        "source": source
    })

    # 7. Contact - Hours (Direct Answer)
    chunks.append({
        "id": "chunk_007",
        "text": f"IAT Networks operation hours are {data['contact_information']['working_hours']}.",
        "section": "contact_information",
        "sub_section": "working_hours",
        "intent": "contact_info",
        "keywords": ["working hours", "office timing", "operation hours", "available hours"],
        "priority": "high",
        "source": source
    })

    # 8. Services - Semantic Split
    for i, service in enumerate(data["services"]):
        sub_list = "\n- ".join(service["sub_services"])
        chunks.append({
            "id": f"chunk_service_{i+1:03d}",
            "text": f"Category: {service['category']}\nDescription: {service['description']}\nServices included:\n- {sub_list}",
            "section": "services",
            "sub_section": service["category"],
            "intent": "service_info",
            "keywords": [service["category"].lower()] + [kw.lower() for kw in service["sub_services"][:3]],
            "priority": "medium",
            "source": source
        })

    # 9. Business Capabilities - Strengths
    chunks.append({
        "id": "chunk_014",
        "text": "IAT Networks core strengths include: " + ", ".join(data["business_capabilities"]["strengths"]),
        "section": "business_capabilities",
        "sub_section": "strengths",
        "intent": "general_info",
        "keywords": ["operational strengths", "competitive advantage", "expert consultation", "scalability"],
        "priority": "medium",
        "source": source
    })

    # 10. Business Capabilities - Values
    chunks.append({
        "id": "chunk_015",
        "text": "Our company values are: " + ", ".join(data["business_capabilities"]["values"]),
        "section": "business_capabilities",
        "sub_section": "values",
        "intent": "company_info",
        "keywords": ["company values", "integrity", "professionalism", "transparency"],
        "priority": "medium",
        "source": source
    })

    # 11. Policies
    chunks.append({
        "id": "chunk_016",
        "text": f"Privacy Policy Summary: {data['policies']['privacy_policy']}",
        "section": "policies",
        "sub_section": "privacy_policy",
        "intent": "policy_info",
        "keywords": ["privacy policy", "data collection", "cookies", "user rights"],
        "priority": "low",
        "source": source
    })

    return chunks

if __name__ == "__main__":
    result = generate_chunks()
    print(json.dumps(result, indent=2))
