"""
Generate 100 Sample Legal QA Dataset

Creates comprehensive dataset with IPC, CrPC, and Constitution entries.
"""

import json

# Legal sections data
legal_data = [
    # IPC Sections (40 entries)
    {"section": "302", "type": "IPC", "topic": "murder", "punishment": "death or life imprisonment"},
    {"section": "304", "type": "IPC", "topic": "culpable homicide", "punishment": "life imprisonment or up to 10 years"},
    {"section": "307", "type": "IPC", "topic": "attempt to murder", "punishment": "up to 10 years"},
    {"section": "376", "type": "IPC", "topic": "rape", "punishment": "rigorous imprisonment not less than 10 years"},
    {"section": "379", "type": "IPC", "topic": "theft", "punishment": "up to 3 years or fine"},
    {"section": "392", "type": "IPC", "topic": "robbery", "punishment": "rigorous imprisonment up to 10 years"},
    {"section": "395", "type": "IPC", "topic": "dacoity", "punishment": "life imprisonment or up to 10 years"},
    {"section": "363", "type": "IPC", "topic": "kidnapping", "punishment": "up to 7 years"},
    {"section": "384", "type": "IPC", "topic": "extortion", "punishment": "up to 3 years or fine"},
    {"section": "420", "type": "IPC", "topic": "cheating", "punishment": "up to 7 years"},
    {"section": "465", "type": "IPC", "topic": "forgery", "punishment": "up to 2 years or fine"},
    {"section": "406", "type": "IPC", "topic": "criminal breach of trust", "punishment": "up to 3 years or fine"},
    {"section": "352", "type": "IPC", "topic": "assault", "punishment": "up to 3 months or fine up to 500 rupees"},
    {"section": "323", "type": "IPC", "topic": "voluntarily causing hurt", "punishment": "up to 1 year or fine up to 1000 rupees"},
    {"section": "325", "type": "IPC", "topic": "voluntarily causing grievous hurt", "punishment": "up to 7 years"},
    {"section": "506", "type": "IPC", "topic": "criminal intimidation", "punishment": "up to 2 years or fine"},
    {"section": "500", "type": "IPC", "topic": "defamation", "punishment": "simple imprisonment up to 2 years or fine"},
    {"section": "447", "type": "IPC", "topic": "criminal trespass", "punishment": "up to 3 months or fine up to 500 rupees"},
    {"section": "448", "type": "IPC", "topic": "house-trespass", "punishment": "up to 1 year or fine up to 1000 rupees"},
    {"section": "426", "type": "IPC", "topic": "mischief", "punishment": "up to 3 months or fine"},
    {"section": "403", "type": "IPC", "topic": "dishonest misappropriation", "punishment": "up to 2 years or fine"},
    {"section": "354", "type": "IPC", "topic": "assault or criminal force to woman", "punishment": "up to 2 years or fine"},
    {"section": "498A", "type": "IPC", "topic": "cruelty by husband", "punishment": "up to 3 years and fine"},
    {"section": "304A", "type": "IPC", "topic": "death by negligence", "punishment": "up to 2 years or fine"},
    {"section": "377", "type": "IPC", "topic": "unnatural offences", "punishment": "life imprisonment or up to 10 years"},
    {"section": "411", "type": "IPC", "topic": "dishonestly receiving stolen property", "punishment": "up to 3 years or fine"},
    {"section": "417", "type": "IPC", "topic": "punishment for cheating", "punishment": "up to 1 year or fine"},
    {"section": "451", "type": "IPC", "topic": "house-trespass in order to commit offence", "punishment": "up to 2 years"},
    {"section": "454", "type": "IPC", "topic": "lurking house-trespass or house-breaking", "punishment": "up to 3 years"},
    {"section": "457", "type": "IPC", "topic": "lurking house-trespass or house-breaking by night", "punishment": "up to 5 years"},
    {"section": "294", "type": "IPC", "topic": "obscene acts and songs", "punishment": "up to 3 months or fine"},
    {"section": "295", "type": "IPC", "topic": "injuring or defiling place of worship", "punishment": "up to 2 years or fine"},
    {"section": "299", "type": "IPC", "topic": "culpable homicide", "definition": "causing death by doing an act with intention or knowledge"},
    {"section": "300", "type": "IPC", "topic": "murder", "definition": "culpable homicide is murder if done with intention to cause death"},
    {"section": "301", "type": "IPC", "topic": "culpable homicide by causing death of wrong person", "definition": "if person causing death intended to kill different person"},
    {"section": "303", "type": "IPC", "topic": "punishment for murder by life-convict", "punishment": "death"},
    {"section": "305", "type": "IPC", "topic": "abetment of suicide of child or insane person", "punishment": "death or life imprisonment"},
    {"section": "306", "type": "IPC", "topic": "abetment of suicide", "punishment": "up to 10 years and fine"},
    {"section": "308", "type": "IPC", "topic": "attempt to commit culpable homicide", "punishment": "up to 3 years or fine"},
    {"section": "309", "type": "IPC", "topic": "attempt to commit suicide", "punishment": "simple imprisonment up to 1 year or fine"},
    {"section": "310", "type": "IPC", "topic": "thug", "definition": "person who is habitually associated with others for purpose of committing robbery or child-stealing"},
    
    # CrPC Sections (30 entries)
    {"section": "41", "type": "CrPC", "topic": "when police may arrest without warrant", "description": "circumstances for arrest without warrant"},
    {"section": "154", "type": "CrPC", "topic": "information in cognizable cases", "description": "procedure for filing FIR"},
    {"section": "156", "type": "CrPC", "topic": "police officer's power to investigate", "description": "investigation of cognizable cases"},
    {"section": "167", "type": "CrPC", "topic": "procedure when investigation cannot be completed", "description": "remand and custody procedures"},
    {"section": "438", "type": "CrPC", "topic": "anticipatory bail", "description": "bail in anticipation of arrest"},
    {"section": "439", "type": "CrPC", "topic": "special powers regarding bail", "description": "High Court or Session Court bail powers"},
    {"section": "437", "type": "CrPC", "topic": "when bail may be taken", "description": "bail in non-bailable cases"},
    {"section": "436", "type": "CrPC", "topic": "bail in bailable offences", "description": "right to bail in bailable cases"},
    {"section": "482", "type": "CrPC", "topic": "inherent powers of High Court", "description": "powers to prevent abuse of process"},
    {"section": "144", "type": "CrPC", "topic": "power to issue order in urgent cases", "description": "temporary orders in urgent cases"},
    {"section": "125", "type": "CrPC", "topic": "order for maintenance of wives, children and parents", "description": "maintenance orders"},
    {"section": "138", "type": "CrPC", "topic": "procedure where he appears to show cause", "description": "show cause proceedings"},
    {"section": "200", "type": "CrPC", "topic": "examination of complainant", "description": "procedure for private complaints"},
    {"section": "202", "type": "CrPC", "topic": "postponement of issue of process", "description": "inquiry before issuing process"},
    {"section": "204", "type": "CrPC", "topic": "issue of process", "description": "summons or warrant issuance"},
    {"section": "205", "type": "CrPC", "topic": "magistrate may dispense with personal attendance", "description": "exemption from personal appearance"},
    {"section": "227", "type": "CrPC", "topic": "discharge", "description": "discharge of accused if no case"},
    {"section": "228", "type": "CrPC", "topic": "framing of charge", "description": "framing charge when prima facie case"},
    {"section": "239", "type": "CrPC", "topic": "when accused shall be discharged", "description": "discharge in warrant cases"},
    {"section": "240", "type": "CrPC", "topic": "framing of charge", "description": "framing charge in warrant cases"},
    {"section": "313", "type": "CrPC", "topic": "power to examine the accused", "description": "examination of accused"},
    {"section": "315", "type": "CrPC", "topic": "accused person to be competent witness", "description": "accused as witness"},
    {"section": "319", "type": "CrPC", "topic": "power to proceed against other persons", "description": "adding accused during trial"},
    {"section": "320", "type": "CrPC", "topic": "compounding of offences", "description": "compounding certain offences"},
    {"section": "321", "type": "CrPC", "topic": "withdrawal from prosecution", "description": "public prosecutor may withdraw"},
    {"section": "353", "type": "CrPC", "topic": "judgment to be pronounced", "description": "pronouncement of judgment"},
    {"section": "354", "type": "CrPC", "topic": "language and contents of judgment", "description": "format of judgment"},
    {"section": "374", "type": "CrPC", "topic": "appeal from convictions", "description": "appeal against conviction"},
    {"section": "397", "type": "CrPC", "topic": "calling for records to exercise powers of revision", "description": "revision powers"},
    {"section": "401", "type": "CrPC", "topic": "High Court's powers of revision", "description": "revision by High Court"},
    
    # Constitution Articles (30 entries)
    {"section": "14", "type": "Constitution", "topic": "equality before law", "description": "no discrimination by State"},
    {"section": "15", "type": "Constitution", "topic": "prohibition of discrimination", "description": "no discrimination on grounds of religion, race, caste, sex"},
    {"section": "19", "type": "Constitution", "topic": "protection of certain rights", "description": "six fundamental freedoms"},
    {"section": "21", "type": "Constitution", "topic": "protection of life and personal liberty", "description": "right to life and liberty"},
    {"section": "32", "type": "Constitution", "topic": "remedies for enforcement of rights", "description": "right to move Supreme Court"},
    {"section": "226", "type": "Constitution", "topic": "power of High Courts to issue writs", "description": "High Court writ jurisdiction"},
    {"section": "25", "type": "Constitution", "topic": "freedom of conscience and free profession", "description": "freedom of religion"},
    {"section": "26", "type": "Constitution", "topic": "freedom to manage religious affairs", "description": "religious denomination rights"},
    {"section": "29", "type": "Constitution", "topic": "protection of interests of minorities", "description": "minority rights"},
    {"section": "30", "type": "Constitution", "topic": "right of minorities to establish educational institutions", "description": "minority educational rights"},
    {"section": "31A", "type": "Constitution", "topic": "saving of laws providing for acquisition of estates", "description": "property acquisition"},
    {"section": "31B", "type": "Constitution", "topic": "validation of certain Acts and Regulations", "description": "Ninth Schedule"},
    {"section": "39A", "type": "Constitution", "topic": "equal justice and free legal aid", "description": "legal aid provision"},
    {"section": "44", "type": "Constitution", "topic": "uniform civil code", "description": "directive principle"},
    {"section": "51A", "type": "Constitution", "topic": "fundamental duties", "description": "duties of citizens"},
    {"section": "124A", "type": "Constitution", "topic": "sedition", "description": "law relating to sedition"},
    {"section": "300A", "type": "Constitution", "topic": "persons not to be deprived of property", "description": "property rights"},
    {"section": "311", "type": "Constitution", "topic": "dismissal, removal or reduction in rank", "description": "civil service protection"},
    {"section": "352", "type": "Constitution", "topic": "proclamation of Emergency", "description": "national emergency"},
    {"section": "356", "type": "Constitution", "topic": "provisions in case of failure of constitutional machinery", "description": "President's rule"},
    {"section": "360", "type": "Constitution", "topic": "provisions as to financial emergency", "description": "financial emergency"},
    {"section": "368", "type": "Constitution", "topic": "power of Parliament to amend the Constitution", "description": "constitutional amendment"},
    {"section": "370", "type": "Constitution", "topic": "temporary provisions with respect to State of Jammu and Kashmir", "description": "special status"},
    {"section": "371", "type": "Constitution", "topic": "special provision with respect to the States", "description": "special state provisions"},
    {"section": "243", "type": "Constitution", "topic": "definition", "description": "Panchayats definition"},
    {"section": "243A", "type": "Constitution", "topic": "Gram Sabha", "description": "village assembly"},
    {"section": "243B", "type": "Constitution", "topic": "constitution of Panchayats", "description": "Panchayat structure"},
    {"section": "243C", "type": "Constitution", "topic": "composition of Panchayats", "description": "Panchayat composition"},
    {"section": "243D", "type": "Constitution", "topic": "reservation of seats", "description": "reservation in Panchayats"},
    {"section": "243E", "type": "Constitution", "topic": "duration of Panchayats", "description": "term of Panchayats"}
]

def generate_qa_entry(legal_item):
    """Generate QA entry for a legal section."""
    section = legal_item["section"]
    section_type = legal_item["type"]
    topic = legal_item.get("topic", "")
    
    # Generate questions
    questions_en = [
        f"What does {section_type} Section {section} say?",
        f"What is {section_type} Section {section}?",
        f"Explain {section_type} Section {section}.",
        f"What does {section_type} Section {section} deal with?",
    ]
    
    questions_hi = [
        f"{section_type} धारा {section} क्या कहती है?",
        f"{section_type} धारा {section} क्या है?",
        f"{section_type} धारा {section} समझाएं।",
    ]
    
    # Generate answers based on type
    if section_type == "IPC":
        answer_en = f"IPC Section {section} deals with {topic}. {legal_item.get('punishment', legal_item.get('definition', 'This section provides legal provisions regarding this matter.'))}"
        answer_hi = f"IPC धारा {section} {topic} से संबंधित है। {legal_item.get('punishment', legal_item.get('definition', 'यह धारा इस मामले से संबंधित कानूनी प्रावधान प्रदान करती है।'))}"
    elif section_type == "CrPC":
        answer_en = f"CrPC Section {section} deals with {topic}. {legal_item.get('description', 'This section provides procedural provisions for criminal cases.')}"
        answer_hi = f"CrPC धारा {section} {topic} से संबंधित है। {legal_item.get('description', 'यह धारा आपराधिक मामलों के लिए प्रक्रियात्मक प्रावधान प्रदान करती है।')}"
    else:  # Constitution
        answer_en = f"Article {section} of the Constitution deals with {topic}. {legal_item.get('description', 'This article provides constitutional provisions.')}"
        answer_hi = f"संविधान का अनुच्छेद {section} {topic} से संबंधित है। {legal_item.get('description', 'यह अनुच्छेद संवैधानिक प्रावधान प्रदान करता है।')}"
    
    context = f"{section_type} Section {section}: {answer_en}"
    
    return {
        "question": questions_en[0],
        "answer": answer_en,
        "context": context,
        "legal_section": f"{section_type} Section {section}",
        "language": "en",
        "section_type": section_type,
        "section_number": section,
        "hindi_question": questions_hi[0],
        "hindi_answer": answer_hi
    }

# Generate all entries
entries = []
for item in legal_data[:100]:  # Take first 100
    entries.append(generate_qa_entry(item))

# Create dataset
dataset = {
    "dataset_name": "Multilingual Legal QA - 100 Sample Dataset",
    "description": "100 comprehensive training entries covering IPC, CrPC, and Constitution with Hindi and English",
    "total_entries": len(entries),
    "entries": entries
}

# Save to file
with open("sample_data_100.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"Generated {len(entries)} sample entries")
print(f"Saved to sample_data_100.json")

