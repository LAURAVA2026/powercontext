# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CSV row Sources with stable identities and lossless text-evidence projection."""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel, JsonValue

from powercontext.sources import AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


class CSVDocumentInput(BaseModel):
    """An offline CSV document and the column that uniquely identifies its rows."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One parsed CSV record, preserving header order and all field strings."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """A captured, independently addressable CSV row."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def _validate_document_id(document_id: str) -> None:
    if not isinstance(document_id, str) or not document_id or document_id != document_id.strip():
        raise ValueError("document_id must be non-empty and have no surrounding whitespace")


def _validate_identity_component(value: str, field: str) -> None:
    if not isinstance(value, str) or not value or not value.strip():
        raise ValueError(f"{field} must be non-empty and not blank")


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Return the delimiter-safe stable identity for one document row."""
    _validate_document_id(document_id)
    _validate_identity_component(primary_key_value, "primary_key_value")
    source_id = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(source_id) > 256:
        raise ValueError("CSV row source identity must not exceed 256 characters")
    return source_id


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse and validate an entire CSV document atomically into row inputs."""
    _validate_document_id(value.document_id)
    if not isinstance(value.primary_key, str) or not value.primary_key:
        raise ValueError("primary_key must be non-empty")
    if not isinstance(value.content, bytes):
        raise ValueError("content must be bytes")

    try:
        text = value.content.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("CSV content must be valid UTF-8") from error

    reader = csv.reader(io.StringIO(text, newline=""), delimiter=",", quotechar='"', doublequote=True, strict=True)
    try:
        try:
            columns_list = next(reader)
        except StopIteration:
            raise ValueError("CSV document must contain a header") from None

        columns = tuple(columns_list)
        if not columns:
            raise ValueError("CSV header must not be empty")
        if any(not column or not column.strip() for column in columns):
            raise ValueError("CSV header fields must be non-empty and not blank")
        if len(set(columns)) != len(columns):
            raise ValueError("CSV header fields must be unique")
        if value.primary_key not in columns:
            raise ValueError("primary_key must exactly match a CSV header field")

        key_index = columns.index(value.primary_key)
        seen_keys: set[str] = set()
        rows: list[CSVRowInput] = []

        for record in reader:
            # csv.reader returns [] for a completely blank physical record.
            if record == []:
                continue
            if len(record) != len(columns):
                raise ValueError("CSV record width must match the header")
            primary_key_value = record[key_index]
            if not primary_key_value or not primary_key_value.strip():
                raise ValueError("CSV primary-key values must be non-empty and not blank")
            if primary_key_value in seen_keys:
                raise ValueError("CSV primary-key values must be unique within a document")
            seen_keys.add(primary_key_value)
            rows.append(
                CSVRowInput(
                    document_id=value.document_id,
                    primary_key=value.primary_key,
                    primary_key_value=primary_key_value,
                    columns=columns,
                    values=tuple(record),
                )
            )
    except csv.Error as error:
        raise ValueError("Malformed CSV document") from error

    # Return only after the complete document has parsed and validated.
    return tuple(rows)


class CSVRowSourceAdapter:
    """Resolve row inputs to captured Sources and read their exact row values."""

    input_class = CSVRowInput
    name = "csv-row"
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        if len(value.columns) != len(value.values):
            raise ValueError("CSV row columns and values must have equal lengths")
        if not value.columns or len(set(value.columns)) != len(value.columns):
            raise ValueError("CSV row columns must be non-empty and unique")
        if value.primary_key not in value.columns:
            raise ValueError("primary_key must exactly match a CSV row column")
        if value.values[value.columns.index(value.primary_key)] != value.primary_key_value:
            raise ValueError("primary_key_value must match the corresponding row value")
        return CSVRowSource(
            name=source_id,
            source_id=source_id,
            definition_version="1",
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Expose the exact row as compact JSON with complete provenance metadata."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row: dict[str, JsonValue] = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type="csv-row",
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_ADAPTER = CSVRowSourceAdapter()
CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(
    CSV_ROW_SOURCE_ADAPTER,
    version="1",
    projections=(CSVRowTextEvidenceProjection(),),
)

__all__ = [
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "CSV_ROW_SOURCE_DEFINITION",
    "csv_row_source_id",
    "parse_csv_document",
]
