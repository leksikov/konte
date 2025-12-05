"""End-to-end tests for CLI module (full workflow with real API calls)."""

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

runner = CliRunner()

# Skip all tests if OPENAI_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@pytest.fixture
def tariff_document(tmp_path):
    """Create a realistic tariff document for testing."""
    doc_path = tmp_path / "tariff_schedule.md"
    doc_path.write_text(
        """
# Korean Tariff Schedule - Chapter 85

## Section XVI: Machinery and Mechanical Appliances

### 8542 - Electronic Integrated Circuits

#### 8542.31 - Processors and Controllers

**Tariff Code: 8542.31.00.10**
- Description: Multi-component integrated circuits (MCOs) - Processors
- Base Rate: 0%
- MFN Rate: 0%
- FTA-Korea Rate: 0%

Processors and controllers, whether or not combined with memories, converters,
logic circuits, amplifiers, clock and timing circuits, or other circuits.

This includes:
- Central processing units (CPUs)
- Microprocessors
- Microcontrollers
- Digital signal processors (DSPs)

#### 8542.32 - Memories

**Tariff Code: 8542.32.00.00**
- Description: Electronic integrated circuits - Memories
- Base Rate: 0%
- MFN Rate: 0%

Memory types covered:
- Dynamic random-access memories (DRAMs)
- Static random-access memories (SRAMs)
- Read-only memories (ROMs, PROMs, EPROMs, EEPROMs)
- Flash memories

### 8541 - Semiconductor Devices

#### 8541.10 - Diodes

**Tariff Code: 8541.10.00.00**
- Description: Diodes, other than photosensitive or light-emitting diodes
- Base Rate: 0%
- MFN Rate: 0%

#### 8541.40 - Photosensitive Devices

**Tariff Code: 8541.40.00.00**
- Description: Photosensitive semiconductor devices
- Base Rate: 3.7%
- MFN Rate: 3.7%

Includes:
- Photovoltaic cells
- Light-emitting diodes (LEDs)
- Laser diodes

## Classification Notes

1. For the purposes of heading 8542, "electronic integrated circuits" means:
   - Monolithic integrated circuits
   - Hybrid integrated circuits
   - Multi-chip integrated circuits

2. Anti-dumping duties may apply to imports from certain countries.

3. Customs valuation follows WTO Agreement on Customs Valuation.
        """
    )
    return doc_path


@pytest.mark.e2e
class TestCLIFullWorkflow:
    """Test complete CLI workflow: create -> add -> build -> query."""

    def test_full_workflow_without_context(self, tmp_path, tariff_document):
        """Test full workflow without context generation (fast)."""
        from konte.cli.app import app

        project_name = "e2e_cli_test"

        # 1. Create project
        result = runner.invoke(app, ["create", project_name, "--storage", str(tmp_path)])
        assert result.exit_code == 0
        assert "Created project" in result.stdout

        # 2. Add document
        result = runner.invoke(app, ["add", project_name, str(tariff_document), "--storage", str(tmp_path)])
        assert result.exit_code == 0
        assert "Added 1 document(s)" in result.stdout

        # 3. Build indexes (skip context for speed)
        result = runner.invoke(app, ["build", project_name, "--skip-context", "--storage", str(tmp_path)])
        assert result.exit_code == 0
        assert "Build complete" in result.stdout

        # 4. Query - semantic
        result = runner.invoke(app, [
            "query", project_name, "What is the tariff code for processors?",
            "--mode", "semantic", "--top-k", "3", "--storage", str(tmp_path)
        ])
        assert result.exit_code == 0
        assert "8542" in result.stdout  # Should find processor tariff code

        # 5. Query - lexical
        result = runner.invoke(app, [
            "query", project_name, "8542.31.00.10",
            "--mode", "lexical", "--top-k", "3", "--storage", str(tmp_path)
        ])
        assert result.exit_code == 0
        assert "Processors" in result.stdout or "processors" in result.stdout

        # 6. Query - hybrid
        result = runner.invoke(app, [
            "query", project_name, "LED semiconductor devices",
            "--mode", "hybrid", "--top-k", "3", "--storage", str(tmp_path)
        ])
        assert result.exit_code == 0
        assert "Results:" in result.stdout

        # 7. Check info
        result = runner.invoke(app, ["info", project_name, "--storage", str(tmp_path)])
        assert result.exit_code == 0
        assert "FAISS Index" in result.stdout
        assert "exists" in result.stdout

        # 8. Delete project
        result = runner.invoke(app, ["delete", project_name, "--force", "--storage", str(tmp_path)])
        assert result.exit_code == 0
        assert "Deleted project" in result.stdout

        # 9. Verify deleted
        result = runner.invoke(app, ["info", project_name, "--storage", str(tmp_path)])
        assert result.exit_code == 1
        assert "not found" in result.stdout


@pytest.mark.e2e
class TestCLIWithContextGeneration:
    """Test CLI workflow with actual context generation."""

    def test_build_with_context_generation(self, tmp_path, tariff_document):
        """Test building with LLM context generation."""
        from konte.cli.app import app

        project_name = "e2e_cli_context"

        # Create and add
        runner.invoke(app, ["create", project_name, "--storage", str(tmp_path)])
        runner.invoke(app, ["add", project_name, str(tariff_document), "--storage", str(tmp_path)])

        # Build WITH context generation (slower, uses LLM)
        result = runner.invoke(app, ["build", project_name, "--storage", str(tmp_path)])
        assert result.exit_code == 0
        assert "Build complete" in result.stdout
        assert "Context generation: enabled" in result.stdout

        # Query should return results with context
        result = runner.invoke(app, [
            "query", project_name, "memory chips tariff",
            "--top-k", "2", "--storage", str(tmp_path)
        ])
        assert result.exit_code == 0
        assert "Context:" in result.stdout  # Context should be shown


@pytest.mark.e2e
class TestCLIPersistence:
    """Test CLI project persistence."""

    def test_project_persists_after_operations(self, tmp_path, tariff_document):
        """Test that project state persists correctly."""
        from konte.cli.app import app

        project_name = "e2e_cli_persist"

        # Create, add, build
        runner.invoke(app, ["create", project_name, "--storage", str(tmp_path)])
        runner.invoke(app, ["add", project_name, str(tariff_document), "--storage", str(tmp_path)])
        runner.invoke(app, ["build", project_name, "--skip-context", "--storage", str(tmp_path)])

        # Query once
        result1 = runner.invoke(app, [
            "query", project_name, "processors",
            "--top-k", "2", "--storage", str(tmp_path)
        ])
        assert result1.exit_code == 0

        # Query again (should work, project was persisted)
        result2 = runner.invoke(app, [
            "query", project_name, "processors",
            "--top-k", "2", "--storage", str(tmp_path)
        ])
        assert result2.exit_code == 0

        # Results should be similar
        assert "Results:" in result1.stdout
        assert "Results:" in result2.stdout


@pytest.mark.e2e
class TestCLIMultipleDocuments:
    """Test CLI with multiple documents."""

    def test_add_multiple_documents_workflow(self, tmp_path, tariff_document):
        """Test adding multiple documents and querying."""
        from konte.cli.app import app

        project_name = "e2e_cli_multi"

        # Create additional document
        doc2 = tmp_path / "customs_rules.md"
        doc2.write_text(
            """
# Customs Valuation Rules

## WTO Agreement on Customs Valuation

The customs value is primarily the transaction value - the price actually paid
or payable for goods when sold for export.

### Methods of Valuation

1. **Transaction Value** - Primary method
2. **Transaction Value of Identical Goods**
3. **Transaction Value of Similar Goods**
4. **Deductive Value**
5. **Computed Value**
6. **Fall-back Method**

### Anti-Dumping Duties

Anti-dumping duties are imposed when:
- Goods are exported at less than normal value
- Material injury to domestic industry is caused
- Causal link exists between dumping and injury
            """
        )

        # Create project
        runner.invoke(app, ["create", project_name, "--storage", str(tmp_path)])

        # Add multiple documents
        result = runner.invoke(app, [
            "add", project_name,
            str(tariff_document), str(doc2),
            "--storage", str(tmp_path)
        ])
        assert result.exit_code == 0
        assert "Added 2 document(s)" in result.stdout

        # Build
        runner.invoke(app, ["build", project_name, "--skip-context", "--storage", str(tmp_path)])

        # Query should find content from both documents
        result = runner.invoke(app, [
            "query", project_name, "anti-dumping duties",
            "--top-k", "5", "--storage", str(tmp_path)
        ])
        assert result.exit_code == 0
        assert "anti-dumping" in result.stdout.lower() or "Anti-dumping" in result.stdout
