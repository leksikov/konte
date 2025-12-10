# Konte Knowledge Base Analysis Report

This report analyzes each Konte project to document their content, structure, and recommended retrieval use cases.

---

## Summary Table

| Project | Chunks | HS Coverage | Language | Best Use Cases |
|---------|--------|-------------|----------|----------------|
| `관세율표_용어_가이드_qwen3` | 8,671 | **All chapters (01-97)** | Korean | Terminology lookup, species identification, comprehensive reference |
| `별표__관세율표(제50조_관련)(관세법)` | 898 | **All chapters (01-97)** | Korean | Official tariff rates, HS code lookup, legal reference |
| `2016년_관세율표_용어_따라잡기_qwen3` | 700 | **All chapters (01-97)** | Korean/English bilingual | Bilingual terminology, consumer-friendly explanations |
| `품목분류_qwen3` | 257 | **Chapters 84-85 only** | Korean | Machinery/electronics classification only |

**Note**: `tariff_legal_framework` project no longer exists in ~/.konte/

---

## 1. 관세율표_용어_가이드_qwen3

### Document Information
- **Title**: 펼쳐보는 관세율표 용어사전 (Comprehensive Tariff Schedule Terminology Dictionary)
- **Publisher**: 관세청 중앙관세분석소 (Korea Customs Service Central Tariff Analysis Office)
- **Publication Date**: November 2019
- **Total Pages**: 3,318 pages
- **Total Chunks**: 8,671 (largest knowledge base)

### Document Structure

**Opening Section**:
- Director's foreword (발刊辭)
- Table of contents listing all 21 sections (제1부-제21부)
- Usage instructions

**Main Content** (organized by HS Section):
- 제1부: Live animals and animal products (Chapters 01-05)
- 제2부: Vegetable products (Chapters 06-14)
- 제3부: Animal/vegetable fats and oils (Chapter 15)
- 제4부: Prepared foodstuffs, beverages, tobacco (Chapters 16-24)
- 제5부: Mineral products (Chapters 25-27)
- 제6부: Chemical products (Chapters 28-38)
- 제7부: Plastics and rubber (Chapters 39-40)
- 제8부: Hides, leather, furs (Chapters 41-43)
- 제9부: Wood products (Chapters 44-46)
- 제10부: Pulp and paper (Chapters 47-49)
- 제11부: Textiles (Chapters 50-63)
- 제12부: Footwear, umbrellas (Chapters 64-67)
- 제13부: Stone, glass, ceramics (Chapters 68-70)
- 제14부: Precious metals and jewelry (Chapter 71)
- 제15부: Base metals (Chapters 72-83)
- 제16부: Machinery (Chapters 84-85)
- 제17부: Vehicles (Chapters 86-89)
- 제18부: Optical, measuring instruments (Chapters 90-92)
- 제19부: Arms and ammunition (Chapter 93)
- 제20부: Miscellaneous articles (Chapters 94-96)
- 제21부: Works of art and antiques (Chapter 97)

**Ending Section**:
- Publishing credits
- Editorial team
- Legal disclaimer: "NOT a classification determination - reference only"

### Content Type
**Official Government Reference Guide** - Combines:
- ~9,600 indexed terminology entries
- Tariff schedule explanatory notes
- Encyclopedia definitions
- Product images and diagrams
- WCO HS classification opinions
- Cross-ministry classification regulations

### Language
**Korean** with English terms in parentheses (scientific names, trade terminology)

### Recommended Retrieval Use Cases
| Query Type | Effectiveness | Example Queries |
|------------|---------------|-----------------|
| Species/taxonomy identification | ⭐⭐⭐⭐⭐ | "Octopus aegina", "Amphioctopus species" |
| Terminology definitions | ⭐⭐⭐⭐⭐ | "냉동 정의", "조제 preserved 의미" |
| Product material lookup | ⭐⭐⭐⭐⭐ | "폴리에스터 섬유", "스테인리스강 종류" |
| HS heading explanations | ⭐⭐⭐⭐ | "제3류 어류", "제16류 조제식품" |
| Classification rules | ⭐⭐⭐ | "GRI 해석규칙" (limited depth) |

**Best For**: Terminology lookup when you need to understand what a term means or identify a species/material before classification.

---

## 2. 별표__관세율표(제50조_관련)(관세법)

### Document Information
- **Title**: [별표] 관세율표(제50조 관련) - Official Tariff Schedule Annex
- **Legal Reference**: Article 50 of Korean Customs Act (관세법 제50조)
- **Source File**: `[별표 ] 관세율표(제50조 관련)(관세법).md`
- **Total Chunks**: 898

### Document Structure

**Beginning (Chapters 01-05)**:
- HS 0105: Live poultry (chickens, ducks, geese, turkeys)
- HS 0106: Other live animals (mammals, reptiles, insects)
- HS 0201-0210: Meat and edible meat offal
- HS 0301-0307: Fish, crustaceans, molluscs

**Middle (Chapters 06-95)**:
- Complete HS classification hierarchy
- Each entry includes: HS code, product description, tariff rate (%)
- Detailed notes (주) explaining scope and exceptions

**End (Chapters 96-97)**:
- HS 9619: Hygiene products (sanitary items)
- HS 9620: Umbrellas, walking sticks
- HS 9701-9706: Works of art, antiques, collectibles (mostly duty-free)

### Content Type
**Official Legal Document** - The authoritative Korean tariff schedule containing:
- HS code classifications (2-10 digit codes)
- Tariff duty rates (무세, 3%, 8%, 18%, 20%, 22.5%, 25%, 27%, 30%)
- Classification notes explaining inclusions/exclusions
- Scientific names for biological products

### Language
**Korean** with Latin scientific nomenclature

### Recommended Retrieval Use Cases
| Query Type | Effectiveness | Example Queries |
|------------|---------------|-----------------|
| Tariff rate lookup | ⭐⭐⭐⭐⭐ | "문어 관세율", "0307.52 세율" |
| HS code search | ⭐⭐⭐⭐⭐ | "8471 컴퓨터", "6204 여성의류" |
| Product classification | ⭐⭐⭐⭐ | "냉동 새우 분류", "LED 조명 HS코드" |
| Tariff notes/exceptions | ⭐⭐⭐⭐ | "제3류 주 해설" |
| Classification rules (GRI) | ⭐⭐ | Limited - has codes but not decision rules |

**Best For**: Finding actual HS codes and tariff rates. Use when you need the official classification code or duty percentage.

---

## 3. 2016년_관세율표_용어_따라잡기_qwen3

### Document Information
- **Title**: 관세율표 용어 따라잡기 (Catching Up with Tariff Terminology)
- **Publisher**: 관세청 고객지원센터 (Korea Customs Service Customer Support Center)
- **Publication Number**: 11-1220000-000384-01
- **Publication Year**: 2016
- **Total Entries**: 1,761 terms
- **Total Chunks**: 700

### Document Structure

**Part I: Terminology Dictionary (Pages 11-214)**
- Organized by HS Chapter (류)
- Each entry includes:
  - Korean name (국문)
  - English name (영문)
  - Definition with source attribution
  - Related images/photos
  - Related products [관련 물품]

**Part II: Bilingual Indexes (Pages 217-280)**
- Korean index (국문색인): Pages 217-248
- English index (영문색인): Pages 249-280

### Content Type
**Consumer-Friendly Guidebook** - Designed for:
- Customs brokers
- Import/export businesses
- **Online shoppers (해외직구)** - individuals buying from overseas

### Language
**Bilingual: Korean-English**
- Primary content in Korean
- English equivalents for all terms
- Bilingual indexes for cross-reference

### Recommended Retrieval Use Cases
| Query Type | Effectiveness | Example Queries |
|------------|---------------|-----------------|
| English-Korean term lookup | ⭐⭐⭐⭐⭐ | "frozen octopus 한국어", "mollusc 번역" |
| Consumer product classification | ⭐⭐⭐⭐ | "해외직구 화장품 분류", "전자제품 관세" |
| Basic terminology | ⭐⭐⭐⭐ | "조제식품 정의", "냉동 vs 냉장" |
| Detailed classification rules | ⭐⭐ | Limited depth |
| Species taxonomy | ⭐⭐ | Less detailed than 관세율표_용어_가이드 |

**Best For**: Bilingual queries, consumer-oriented questions, when you need Korean-English translation of tariff terms.

**Important Disclaimer**: Document explicitly states it is "reference only (참고용)" with "no legal force (법률적 효력이 없음)"

---

## 4. 품목분류_qwen3

### Document Information
- **Title**: 누구나 알기 쉬운 품목분류 e-Guide Book (Everyone's Easy-to-Understand Classification e-Guide)
- **Subtitle**: 기계 및 전자기기편 (Machinery & Electronics Edition)
- **Publisher**: 관세평가분류원 (Korea Customs Classification and Valuation Service)
- **Publication Year**: 2004
- **Total Pages**: 130
- **Total Chunks**: 257 (smallest knowledge base)

### Document Structure

**Section 1 (Pages 1-6)**: HS의 개념 (HS System Concepts)
- History and structure of Harmonized System
- International standards (WCO)

**Section 2 (Pages 6-8)**: 관세율표의 法源 (Legal Basis)
- Korean tariff law framework
- Relationship to international conventions

**Section 3 (Pages 12-32)**: 제16부 개요 (Section XVI Overview)
- Introduction to machinery/electronics classification
- General principles

**Section 4 (Pages 15-32)**: 기계와 부분품 (Machinery & Parts)
- Classification concepts
- Parts vs. accessories rules

**Section 5 (Pages 33-84)**: HS 84류 - 기계류 (Chapter 84 - Machinery)
- Detailed classification rules for:
  - Motors, engines, turbines (8401-8412)
  - Pumps, compressors (8413-8414)
  - Industrial machinery (8415-8479)
  - Machine parts (8480-8487)

**Section 6 (Pages 86-130)**: HS 85류 - 전기기기 (Chapter 85 - Electrical Equipment)
- Detailed classification rules for:
  - Electric motors, generators (8501-8504)
  - Batteries, accumulators (8506-8507)
  - Electrical heating/lighting (8509-8516)
  - Telecommunications (8517-8525)
  - Semiconductors (8541-8542)

### Content Type
**Specialized Training Manual** - Deep classification guidance for:
- HS Chapter 84 (Machinery)
- HS Chapter 85 (Electrical Equipment)
- GRI (General Rules of Interpretation) application
- Parts and accessories classification

### Language
**Korean** with English technical terminology (HS, GRI, WCO, etc.)

### HS Code Coverage
**LIMITED: Chapters 84-85 ONLY**

| Covered | Not Covered |
|---------|-------------|
| HS 84 (Machinery) | Chapters 01-83 |
| HS 85 (Electrical) | Chapters 86-97 |

### Recommended Retrieval Use Cases
| Query Type | Effectiveness | Example Queries |
|------------|---------------|-----------------|
| Machinery classification | ⭐⭐⭐⭐⭐ | "펌프 분류", "84류 기계 부분품" |
| Electronics classification | ⭐⭐⭐⭐⭐ | "반도체 HS코드", "LED 조명기기" |
| Parts vs. accessories rules | ⭐⭐⭐⭐⭐ | "부분품 분류원칙", "범용 부분품" |
| GRI rules for machinery | ⭐⭐⭐⭐⭐ | "기능단위 분류", "미완성 기계" |
| Food/agricultural products | ❌ | NOT APPLICABLE |
| Textiles/chemicals | ❌ | NOT APPLICABLE |
| Seafood (0307) | ❌ | NOT APPLICABLE |

**Best For**: Classifying machinery, electronics, and their parts. Contains detailed decision rules and GRI application specific to Chapters 84-85.

**DO NOT USE FOR**: Food, animals, textiles, chemicals, or any product outside HS 84-85.

---

## Retrieval Strategy by Product Type

### Seafood (HS 03, 16)
1. **First**: `관세율표_용어_가이드_qwen3` - Species identification
2. **Second**: `별표__관세율표` - HS code and tariff rate
3. **Avoid**: `품목분류_qwen3` (no relevant content)

### Machinery/Electronics (HS 84-85)
1. **First**: `품목분류_qwen3` - Detailed classification rules
2. **Second**: `별표__관세율표` - HS code and tariff rate
3. **Third**: `관세율표_용어_가이드_qwen3` - Technical terminology

### Consumer Products / 해외직구
1. **First**: `2016년_관세율표_용어_따라잡기_qwen3` - Consumer-friendly, bilingual
2. **Second**: `별표__관세율표` - Official rates

### Bilingual Queries (Korean ↔ English)
1. **Only**: `2016년_관세율표_용어_따라잡기_qwen3` - Has English index

### Official Tariff Rates
1. **Only**: `별표__관세율표` - Legal/binding rates

---

## Missing Project: tariff_legal_framework

The project `tariff_legal_framework` referenced in previous testing **no longer exists** in ~/.konte/.

Based on earlier test results, it contained:
- WCO HS explanatory notes
- GRI (General Rules of Interpretation)
- Chapter vs. heading classification rules
- 4,363 chunks

**Recommendation**: Rebuild this project if GRI classification rules are needed. Currently, no existing project provides deep GRI guidance for non-machinery products.
