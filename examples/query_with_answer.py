#!/usr/bin/env python3
"""Example: Full RAG pipeline - retrieval + LLM answer generation.

Demonstrates:
- query_with_answer() for end-to-end RAG
- Custom prompt templates
- GeneratedAnswer model (answer text, sources_used, model)
- Suggested action hints for agents
"""

import asyncio
from pathlib import Path

from konte import Project, delete_project


SAMPLE_CONTENT = """
# HS Code Classification Guide: Chapter 85

## 8542 - Electronic Integrated Circuits

### 8542.31 - Processors and Controllers
Microprocessors, microcontrollers, and digital signal processors.
Classification requires the component to be in monolithic integrated circuit form.
Examples: Intel Core i9, AMD Ryzen 9, Apple M3, ARM Cortex-A78.

### 8542.32 - Memories
Dynamic RAM (DRAM), static RAM (SRAM), flash memory, EEPROM.
Must be semiconductor-based storage in integrated circuit form.
Examples: Samsung DDR5 RAM, SK Hynix NAND flash, Micron 3D NAND.

### 8542.33 - Amplifiers
Operational amplifiers, power amplifiers, audio amplifiers.
Classified here when in monolithic IC form.

### 8542.39 - Other Integrated Circuits
ASICs, FPGAs, mixed-signal ICs, and other specialized circuits.
Includes gate arrays, programmable logic devices.

## 8471 - Automatic Data Processing Machines (Computers)

### 8471.30 - Portable Computers (Laptops)
Weighing not more than 10 kg, with display, keyboard, and battery.
Includes laptops, notebooks, and tablets with keyboard.

### 8471.41 - Other Computers with Processing Unit
Desktop computers, workstations, servers.
Classification based on primary function as data processing machine.

## 8523 - Storage Media

### 8523.51 - Solid-State Non-Volatile Storage
USB flash drives, solid-state drives (SSD), memory cards.
Classified here when packaged as complete storage device (not bare IC).
Key distinction from 8542.32: complete storage unit vs bare memory chip.
"""


async def main():
    storage_path = Path("/tmp/konte_answer_example")
    project_name = "hs_chapter85"

    # --- Setup ---
    print("Building project...")
    project = Project.create(name=project_name, storage_path=storage_path)

    content_file = storage_path / project_name / "chapter85.md"
    content_file.parent.mkdir(parents=True, exist_ok=True)
    content_file.write_text(SAMPLE_CONTENT, encoding="utf-8")

    project.add_documents([content_file])
    await project.build(skip_context=True)
    project.save()
    print("Project ready.\n")

    # --- 1. Basic query_with_answer ---
    print("=" * 60)
    print("1. Basic query_with_answer()")
    print("=" * 60)

    query = "What HS code should I use for DDR5 RAM chips?"
    response, answer = await project.query_with_answer(query=query, max_chunks=5)

    print(f"Query: {query}")
    print(f"Answer: {answer.answer}")
    print(f"Sources used: {answer.sources_used}")
    print(f"Model: {answer.model}")
    print(f"Top retrieval score: {response.top_score:.2f}")
    print(f"Suggested action: {response.suggested_action}")

    # --- 2. Custom prompt template ---
    print("\n" + "=" * 60)
    print("2. Custom prompt template")
    print("=" * 60)

    custom_prompt = """Based on the following reference documents, provide a classification decision.

Reference documents:
{context}

Classification request: {question}

Provide:
1. Recommended HS code
2. Brief justification
3. Key distinguishing factors

Decision:"""

    query = "How to classify a USB flash drive vs a bare NAND flash chip?"
    response, answer = await project.query_with_answer(
        query=query,
        max_chunks=5,
        prompt_template=custom_prompt,
    )

    print(f"Query: {query}")
    print(f"Answer: {answer.answer}")

    # --- 3. Retrieval response inspection ---
    print("\n" + "=" * 60)
    print("3. Inspecting retrieval results alongside answer")
    print("=" * 60)

    query = "What is classified under 8542.39?"
    response, answer = await project.query_with_answer(query=query, max_chunks=3)

    print(f"Query: {query}")
    print(f"\nRetrieved {response.total_found} chunks:")
    for i, result in enumerate(response.results[:3], 1):
        print(f"  [{i}] score={result.score:.2f} | {result.content[:80]}...")

    print(f"\nGenerated answer: {answer.answer}")

    # --- Cleanup ---
    delete_project(project_name, storage_path=storage_path)
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
