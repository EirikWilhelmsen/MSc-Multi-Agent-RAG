from help_functions import call_llm, format_chunks, parse_scores
import json
import re

def generation_agent(query: str, chunk: str) -> tuple[str, int]:
    prompt = (
        f"Given a question and some relevant documents, generate a SHORT ANSWER "
        f"to the question based on the documents.\n\n"
        f"# Question\n{query}\n\n"
        f"# Text\n{chunk}\n\n"
        f"# Requirements\n"
        f"- Please give a SHORT ANSWER. Use as few words as possible.\n"
        f"- If the answer is a number with more than 4 digits, use commas as thousand separators (e.g., \"1,000\" instead of \"1000\").\n"
        f"- Don't include period at the end of the answer.\n"
        f"- If you are not sure about the answer, you MUST reply \"Unsure\".\n"
        f"in the documents.\n\n"
        f"# Answer"
    )
    response, tokens = call_llm(prompt, model="gorina10.llama3.3:70b")
    return response, tokens

def relevance_agent(query: str, chunks: list[str]) -> tuple[str, float]:
    prompt = f"""You are a relevance ranking agent.
        Given the question and a list of text chunks, rank each chunk by how
        directly and completely it answers the question.
        
        Question: {query}
        
        Chunks:
        {format_chunks(chunks)}
        
        For each chunk, output a relevance score from 0.0 to 1.0.
        Think step by step before scoring.
        Output ONLY valid JSON, no extra text:
        [{{"chunk_id": 0, "score": 0.9, "reasoning": "..."}}, ...]"""

    response, tokens = call_llm(prompt, model="gorina10.llama3.3:70b")
    return parse_scores(response), tokens

def temporal_agent(query: str, chunks: list[str]) -> tuple[dict[int, float], int]:
    prompt = f"""You are a temporal reasoning agent.
        Given a question and text chunks that may contain conflicting information,
        estimate which chunk contains the most RECENT information.
        
        Reason about the content of the text itself, including:
        - Explicit dates or years mentioned in the text
        - Language cues suggesting updates or changes ("now", "currently", 
          "recently", "replaced", "formerly", "previously")
        - Whether one chunk's facts imply the other is outdated
        
        Question: {query}
        
        Chunks:
        {format_chunks(chunks)}
        
        You MUST score ALL {len(chunks)} chunks (chunk_id 0 through {len(chunks)-1}).
        
        Think step by step. Output ONLY valid JSON, no extra text:
        [{{"chunk_id": 0, "recency_score": 0.8, "reasoning": "..."}}, ...]"""

    response, tokens = call_llm(prompt, model="gorina10.llama3.3:70b")
    return parse_scores(response), tokens

def candidate_agent(query: str, chunk: str, chunk_id: int) -> dict:
    prompt = (
        f"Given a question and a text chunk, answer the question based only "
        f"on the text. Then rate your confidence.\n\n"
        f"# Question\n{query}\n\n"
        f"# Text\n{chunk}\n\n"
        f"# Requirements\n"
        f"- Please give a SHORT ANSWER. Use as few words as possible.\n"
        f"- If the answer is a number with more than 4 digits, use commas as thousand separators.\n"
        f"- Don't include period at the end of the answer.\n"
        f"- If you are not sure about the answer, you MUST reply \"Unsure\" as answer.\n\n"
        f"Output ONLY valid JSON:\n"
        f"{{\"answer\": \"...\", \"confidence\": 0.85, \"reasoning\": \"...\"}}"
    )
    response, tokens = call_llm(prompt, model="gorina10.llama3.3:70b")
    
    cleaned = re.sub(r"```(?:json)?|```", "", response).strip()
    try:
        parsed = json.loads(cleaned)
        return {
            "chunk_id": chunk_id,
            "answer": parsed.get("answer", "Unsure"),
            "confidence": float(parsed.get("confidence", 0.0)),
            "reasoning": parsed.get("reasoning", ""),
            "tokens": tokens
        }
    except (json.JSONDecodeError, KeyError):
        return {"chunk_id": chunk_id, "answer": "Unsure", "confidence": 0.0, "reasoning": "", "tokens": tokens}
    
def debate_round(query: str, candidates: list[dict], top_chunks: list[dict], round_num: int, debate_history: list[list[dict]]) -> list[dict]:
    """Each candidate sees the others' answers and argues for their own."""

    updated = []

    for c in candidates:
        others = [o for o in candidates if o["chunk_id"] != c["chunk_id"]]
        others_text = "\n".join(
            f"- Candidate {o['chunk_id']}: \"{o['answer']}\" "
            f"(confidence {o['confidence']:.2f}, reasoning: {o['reasoning']})"
            for o in others
        )

        prev_rounds_text = ""
        if debate_history:
            for r_i, r_entries in enumerate(debate_history):
                prev_rounds_text += f"\n--- Previous debate round {r_i+1} ---\n"
                for entry in r_entries:
                    prev_rounds_text += (
                        f"Candidate {entry['chunk_id']}: \"{entry['answer']}\" "
                        f"(confidence {entry['confidence']:.2f}) — {entry['reasoning']}\n"
                    )

        chunk_text = top_chunks[c["chunk_id"]]["content"]

        prompt = (
            f"You are Candidate {c['chunk_id']} in a debate about the correct answer "
            f"to a question. You have access to your own source text.\n\n"
            f"# Question\n{query}\n\n"
            f"# Your source text\n{chunk_text}\n\n"
            f"# Your current answer\n\"{c['answer']}\" (confidence {c['confidence']:.2f})\n"
            f"Your reasoning: {c['reasoning']}\n\n"
            f"# Other candidates' answers\n{others_text}\n"
            f"{prev_rounds_text}\n"
            f"# Debate round {round_num}\n"
            f"Consider the other candidates' arguments carefully. They may have "
            f"more recent or more accurate information. You may update your answer "
            f"and confidence, or defend your current position with stronger arguments.\n\n"
            f"Think about:\n"
            f"- Does your source text contain more specific or more recent information?\n"
            f"- Are the other candidates' answers contradicting yours? If so, why might "
            f"your source be more reliable?\n"
            f"- If another candidate has a more convincing argument, you may change your answer.\n\n"
            f"# Requirements\n"
            f"- SHORT ANSWER. Use as few words as possible.\n"
            f"- If the answer is a number with more than 4 digits, use commas as thousand separators.\n"
            f"- Don't include period at the end of the answer.\n\n"
            f"Output ONLY valid JSON:\n"
            f"{{\"answer\": \"...\", \"confidence\": 0.85, \"reasoning\": \"...\"}}"
        )

        response, tokens = call_llm(prompt, model="gorina10.llama3.3:70b")
        cleaned = re.sub(r"```(?:json)?|```", "", response).strip()
        try:
            parsed = json.loads(cleaned)
            updated.append({
                "chunk_id": c["chunk_id"],
                "answer": parsed.get("answer", c["answer"]),
                "confidence": float(parsed.get("confidence", c["confidence"])),
                "reasoning": parsed.get("reasoning", c["reasoning"]),
                "tokens": tokens
            })
        except (json.JSONDecodeError, KeyError):
            updated.append(c)  # keep previous if parsing fails

    return updated

def supervisor_agent(query: str, candidates: list[dict], top_chunks: list[dict], debate_history: list[list[dict]]) -> dict:
    """Reviews the full debate and picks the best answer."""

    debate_log = ""
    # Initial positions
    debate_log += "=== Initial positions ===\n"
    for c in debate_history[0]:
        debate_log += (
            f"Candidate {c['chunk_id']}: "
            f"\"{c['answer']}\" (confidence {c['confidence']:.2f}) — {c['reasoning']}\n"
        )

    # Debate rounds
    for r_i, r_entries in enumerate(debate_history[1:], start=1):
        debate_log += f"\n=== Debate round {r_i} ===\n"
        for entry in r_entries:
            debate_log += (
                f"Candidate {entry['chunk_id']}: \"{entry['answer']}\" "
                f"(confidence {entry['confidence']:.2f}) — {entry['reasoning']}\n"
            )

    # Final positions
    debate_log += "\n=== Final positions ===\n"
    for c in candidates:
        debate_log += (
            f"Candidate {c['chunk_id']}: "
            f"\"{c['answer']}\" (confidence {c['confidence']:.2f}) — {c['reasoning']}\n"
        )

    prompt = (
        f"You are a supervisor agent. Your job is to review a debate between "
        f"candidates who each answered a question based on different source texts.\n\n"
        f"# Question\n{query}\n\n"
        f"# Debate transcript\n{debate_log}\n"
        f"# Your task\n"
        f"Select the best answer. Consider:\n"
        f"1. Quality of reasoning and evidence from the source text\n"
        f"2. How well the candidate defended their position during debate\n"
        f"3. Confidence levels and whether they were justified\n\n"
        f"# Requirements\n"
        f"- Pick the winning candidate's answer.\n"
        f"- SHORT ANSWER. Use as few words as possible.\n"
        f"- If the answer is a number with more than 4 digits, use commas as thousand separators.\n"
        f"- Don't include period at the end of the answer.\n\n"
        f"Output ONLY valid JSON:\n"
        f"{{\"winner_chunk_id\": 0, \"answer\": \"...\", \"reasoning\": \"...\"}}"
    )

    response, tokens = call_llm(prompt, model="gorina10.llama3.3:70b")
    cleaned = re.sub(r"```(?:json)?|```", "", response).strip()
    try:
        parsed = json.loads(cleaned)
        winner_id = int(parsed.get("winner_chunk_id", 0))
        return {
            "winner_chunk_id": winner_id,
            "answer": parsed.get("answer", candidates[0]["answer"]),
            "reasoning": parsed.get("reasoning", ""),
            "tokens": tokens
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback: pick highest confidence
        best = max(candidates, key=lambda x: x["confidence"])
        return {
            "winner_chunk_id": best["chunk_id"],
            "answer": best["answer"],
            "reasoning": "Fallback: highest confidence",
            "tokens": tokens
        }
        
def verification_agent(query: str, conflict_info: dict, chunks: list[str]) -> tuple[dict, int]:
    """Utfordrer det foreløpige svaret med oppfølgingsspørsmål basert på konflikten."""
    chunks_str = format_chunks(chunks)
    
    prompt = (
        f"You are a verification agent. A previous agent analyzed these chunks and "
        f"detected a potential conflict.\n\n"
        f"# Original Question\n{query}\n\n"
        f"# Conflict Analysis\n"
        f"- Conflict: {conflict_info.get('conflict_formulation', 'None detected')}\n"
        f"- Hypothesis about outdated info: {conflict_info.get('outdated_hypothesis', 'N/A')}\n"
        f"- Preliminary answer: {conflict_info.get('preliminary_answer', 'Unsure')}\n"
        f"- Source chunk: {conflict_info.get('source_chunk_id', 'Unknown')}\n"
        f"- Confidence: {conflict_info.get('confidence', 0.0)}\n\n"
        f"# Text Chunks\n{chunks_str}\n\n"
        f"# Your Task\n"
        f"1. Generate a follow-up question that would help distinguish which chunk is current.\n"
        f"2. Answer your own follow-up question using evidence from the chunks.\n"
        f"3. Based on this analysis, either CONFIRM or REVISE the source chunk.\n\n"
        f"Output ONLY valid JSON:\n"
        f"{{\n"
        f"  \"followup_question\": \"...\",\n"
        f"  \"followup_reasoning\": \"...\",\n"
        f"  \"decision\": \"confirm\" or \"revise\",\n"
        f"  \"final_source_chunk_id\": 0\n"
        f"}}"
    )
    #print(prompt)
    response, tokens = call_llm(prompt, model="gorina10.llama3.3:70b")
    
    cleaned = re.sub(r"```(?:json)?|```", "", response).strip()
    try:
        return json.loads(cleaned), tokens
    except json.JSONDecodeError:
        # Fallback: behold preliminary answer
        return {
            "followup_question": "",
            "followup_reasoning": "",
            "decision": "confirm",
            "final_source_chunk_id": conflict_info.get("source_chunk_id")
        }, tokens
        
def conflict_detection_agent(query: str, chunks: list[str]) -> tuple[dict, int]:
    """Identifiserer konflikt og formulerer den eksplisitt."""
    chunks_str = format_chunks(chunks)
    prompt = (
        f"You are a conflict detection agent analyzing text chunks that may contain "
        f"contradictory information about the same topic.\n\n"
        f"# Question\n{query}\n\n"
        f"# Text Chunks\n{chunks_str}\n\n"
        f"# Task\n"
        f"1. Identify if chunks contain CONFLICTING answers to the question.\n"
        f"2. If conflict exists, formulate it precisely: what does each chunk claim?\n"
        f"3. Hypothesize which chunk might be outdated vs updated based on:\n"
        f"   - Language suggesting change ('now', 'currently', 'formerly', 'previously', 'renamed')\n"
        f"   - One fact superseding another (e.g., new name vs old name)\n"
        f"   - References to updates or corrections\n"
        f"4. Provide your best answer based on the chunk you believe is most current.\n\n"
        f"# Requirements for the answer\n"
        f"- Please give a SHORT ANSWER. Use as few words as possible.\n"
        f"- If the answer is a number with more than 4 digits, use commas as thousand separators (e.g., \"1,000\" instead of \"1000\").\n"
        f"- Don't include period at the end of the answer.\n"
        f"- If you are not sure about the answer, you MUST reply \"Unsure\".\n"
        f"Output ONLY valid JSON:\n"
        f"{{\n"
        f"  \"conflict_detected\": true/false,\n"
        f"  \"conflict_formulation\": \"Chunk X claims [A], while Chunk Y claims [B] and Chunk Z claims [C]\",\n"
        f"  \"outdated_hypothesis\": \"Chunk X appears outdated because...\",\n"
        f"  \"preliminary_answer\": \"...\",\n"
        f"  \"source_chunk_id\": 0,\n"
        f"  \"confidence\": 0.8\n"
        f"}}"
    )
    #print(prompt)
    response, tokens = call_llm(prompt, model="gorina10.llama3.3:70b")
    
    cleaned = re.sub(r"```(?:json)?|```", "", response).strip()
    try:
        return json.loads(cleaned), tokens
    except json.JSONDecodeError:
        return {
            "conflict_detected": False,
            "conflict_formulation": "",
            "outdated_hypothesis": "",
            "preliminary_answer": "Unsure",
            "source_chunk_id": None,
            "confidence": 0.0
        }, tokens
        
def baseline_generation_agent(query: str, documents: list[dict]) -> tuple[str, int]:
    doc_texts = []
    for i, doc in enumerate(documents, 1):
        doc_texts.append(
            f"## Document {i}\n"
            f"{doc['content']}"
        )

    documents_str = "\n\n".join(doc_texts)
    
    prompt = (
        f"Given a question and some relevant documents, generate a SHORT ANSWER "
        f"to the question based on the document.\n\n" 
        f"# Question\n{query}\n\n"
        f"# Text\n{documents_str}\n\n"
        f"# Requirements\n"
        f"- Please give a SHORT ANSWER. Use as few words as possible.\n"
        f"- If the answer is a number with more than 4 digits, use commas as thousand separators (e.g., \"1,000\" instead of \"1000\").\n"
        f"- Don't include period at the end of the answer.\n"
        f"- If you are not sure about the answer, you MUST reply \"Unsure\".\n"
        f"in the documents.\n\n"
        f"# Answer"
    )
    return call_llm(prompt, model="gorina10.llama3.3:70b")