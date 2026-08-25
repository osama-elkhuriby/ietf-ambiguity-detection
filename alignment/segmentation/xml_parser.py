"""Parser for XML2RFC v2/v3 Internet-Draft source."""

from __future__ import annotations

import re
from pathlib import Path

from lxml import etree  # type: ignore[import-untyped]

from .models import (
    Block,
    Document,
    Reference,
    ReferenceDefinition,
    Section,
    SourceLocation,
)
from .normalize import (
    block_identifier,
    file_sha256,
    normalize_multiline,
    normalize_prose,
)


def _local_name(element: etree._Element) -> str:
    return etree.QName(element).localname


def _children(element: etree._Element, name: str) -> list[etree._Element]:
    return [child for child in element if isinstance(child.tag, str) and _local_name(child) == name]


def _first_child(element: etree._Element, name: str) -> etree._Element | None:
    return next(iter(_children(element, name)), None)


def _element_text(element: etree._Element, multiline: bool = False) -> str:
    text = _inline_text(element)
    return normalize_multiline(text) if multiline else normalize_prose(text)


def _reference_placeholder(element: etree._Element) -> str:
    """Return the text XML2RFC would visibly derive for an empty reference."""
    if element.get("format") == "none":
        return ""

    derived_content = element.get("derivedContent")
    target = element.get("target") or ""
    section = element.get("section")
    name = _local_name(element)
    if derived_content:
        return derived_content
    if name == "eref":
        return target
    if section and target:
        return f"Section {section} of [{target}]"
    if target:
        return f"[{target}]"
    return ""


def _inline_text(element: etree._Element) -> str:
    """Render inline XML while retaining the visible form of empty references."""
    parts = [element.text or ""]
    for child in element:
        if not isinstance(child.tag, str):
            parts.append(child.text or "")
            parts.append(child.tail or "")
            continue

        name = _local_name(child)
        child_text = _inline_text(child)
        if name in {"xref", "relref", "eref"} and not child_text.strip():
            child_text = _reference_placeholder(child)
        parts.append(child_text)
        parts.append(child.tail or "")
    return "".join(parts)


def _xpath(tree: etree._ElementTree, element: etree._Element) -> str:
    return tree.getpath(element)


def _extract_xml_references(element: etree._Element) -> list[Reference]:
    references: list[Reference] = []
    for descendant in element.iter():
        if not isinstance(descendant.tag, str):
            continue
        name = _local_name(descendant)
        if name not in {"xref", "relref", "eref"}:
            continue
        target = descendant.get("target")
        if not target:
            continue
        references.append(
            Reference(
                target=target,
                target_section=descendant.get("section"),
                kind=name,  # type: ignore[arg-type]
                display_text=(
                    _element_text(descendant) or _reference_placeholder(descendant) or None
                ),
            )
        )
    return references


def _text_without_children(element: etree._Element, excluded: set[str]) -> str:
    parts = [element.text or ""]
    for child in element:
        if not isinstance(child.tag, str):
            continue
        if _local_name(child) not in excluded:
            parts.append(_inline_text(child))
        parts.append(child.tail or "")
    return normalize_prose("".join(parts))


def _text_without_descendants(element: etree._Element, excluded: set[str]) -> str:
    """Render an element while omitting nested structural containers."""

    def render(current: etree._Element) -> str:
        parts = [current.text or ""]
        for child in current:
            if not isinstance(child.tag, str):
                parts.append(child.text or "")
            elif _local_name(child) not in excluded:
                child_text = render(child)
                if _local_name(child) in {"xref", "relref", "eref"} and not child_text.strip():
                    child_text = _reference_placeholder(child)
                parts.append(child_text)
            parts.append(child.tail or "")
        return "".join(parts)

    return normalize_prose(render(element))


def _table_text(element: etree._Element) -> str:
    """Render XML2RFC v2 and v3 tables without collapsing cell boundaries."""
    name = _local_name(element)
    rows: list[str] = []
    if name == "table":
        for row in element.iterdescendants():
            if not isinstance(row.tag, str) or _local_name(row) != "tr":
                continue
            cells = [
                _element_text(cell)
                for cell in row
                if isinstance(cell.tag, str) and _local_name(cell) in {"th", "td"}
            ]
            if cells:
                rows.append(" | ".join(cells))
    else:
        headings = [_element_text(column) for column in _children(element, "ttcol")]
        cells = [_element_text(cell) for cell in _children(element, "c")]
        if headings:
            rows.append(" | ".join(headings))
            width = len(headings)
            rows.extend(
                " | ".join(cells[index : index + width])
                for index in range(0, len(cells), width)
            )
        elif cells:
            rows.extend(cells)
    return normalize_multiline("\n".join(rows))


def _block_elements(
    section: etree._Element,
) -> list[tuple[etree._Element, str, str | None]]:
    result: list[tuple[etree._Element, str, str | None]] = []

    def add_nested_lists(container: etree._Element) -> None:
        for nested in container:
            if not isinstance(nested.tag, str):
                continue
            if _local_name(nested) in {"ul", "ol", "dl", "list"}:
                add_element(nested)
            else:
                add_nested_lists(nested)

    def add_list_item(item: etree._Element) -> None:
        item_text = _text_without_descendants(item, {"ul", "ol", "dl", "list"})
        if item_text:
            result.append((item, "list_item", item_text))
        add_nested_lists(item)

    def add_element(child: etree._Element) -> None:
        if not isinstance(child.tag, str):
            return
        name = _local_name(child)
        if name == "t":
            # XML2RFC v2 nests <list> elements inside <t>.  Keep any
            # introductory prose, then emit each nested <t> as a list item.
            legacy_lists = _children(child, "list")
            if not legacy_lists:
                result.append((child, "paragraph", None))
                return
            surrounding_text = _text_without_children(child, {"list"})
            if surrounding_text:
                result.append((child, "paragraph", surrounding_text))
            for legacy_list in legacy_lists:
                for item in _children(legacy_list, "t"):
                    item_text = _text_without_descendants(item, {"list"})
                    hanging_term = item.get("hangText")
                    if hanging_term:
                        item_text = f"{normalize_prose(hanging_term)}: {item_text}"
                    result.append((item, "list_item", item_text))
                    add_nested_lists(item)
        elif name == "list":
            for item in _children(child, "t"):
                add_list_item(item)
        elif name in {"ul", "ol"}:
            for item in _children(child, "li"):
                add_list_item(item)
        elif name == "dl":
            terms = _children(child, "dt")
            definitions = _children(child, "dd")
            for index, definition in enumerate(definitions):
                term = _element_text(terms[index]) if index < len(terms) else ""
                definition_text = _element_text(definition)
                text = f"{term}: {definition_text}" if term else definition_text
                result.append((definition, "definition", text))
        elif name in {"artwork", "sourcecode"}:
            result.append((child, name, None))
        elif name in {"table", "texttable"}:
            result.append((child, "table", _table_text(child)))
        elif name == "figure":
            for nested in child.iterdescendants():
                if not isinstance(nested.tag, str):
                    continue
                nested_name = _local_name(nested)
                if nested_name in {"preamble", "postamble"}:
                    result.append((nested, "paragraph", None))
                elif nested_name in {"artwork", "sourcecode"}:
                    result.append((nested, nested_name, None))
        elif name == "blockquote":
            for nested in child:
                if not isinstance(nested.tag, str):
                    continue
                nested_name = _local_name(nested)
                if nested_name == "t":
                    result.append((nested, "blockquote", None))
                elif nested_name in {"artwork", "sourcecode", "table"}:
                    add_element(nested)
                elif nested_name in {"ul", "ol", "list"}:
                    for item_name in ("li", "t"):
                        for item in _children(nested, item_name):
                            result.append((item, "blockquote", _element_text(item)))
        elif name == "aside":
            for nested in child:
                add_element(nested)

    for child in section:
        add_element(child)
    return result


def _make_blocks(
    tree: etree._ElementTree,
    element: etree._Element,
    document_id: str,
    section_id: str,
    counter: list[int],
) -> list[Block]:
    blocks: list[Block] = []
    for block_element, block_type, supplied_text in _block_elements(element):
        multiline = block_type in {"artwork", "sourcecode", "table"}
        text = (
            supplied_text
            if supplied_text is not None
            else _element_text(block_element, multiline=multiline)
        )
        if not text:
            continue
        counter[0] += 1
        blocks.append(
            Block(
                block_id=block_identifier(document_id, section_id, counter[0]),
                index=counter[0],
                block_type=block_type,  # type: ignore[arg-type]
                text=text,
                source_location=SourceLocation(xpath=_xpath(tree, block_element)),
                references=_extract_xml_references(block_element),
            )
        )
    return blocks


def _parse_section(
    tree: etree._ElementTree,
    element: etree._Element,
    document_id: str,
    section_id: str,
    parent_path: list[str],
    counter: list[int],
    analysis_scope: str = "body",
) -> Section:
    name_element = _first_child(element, "name")
    title = _element_text(name_element) if name_element is not None else element.get("title", "")
    resolved_scope = _body_section_scope(title, analysis_scope)
    section = Section(
        section_id=section_id,
        title=title or "Untitled",
        path=[*parent_path, section_id],
        anchor=element.get("anchor"),
        analysis_scope=resolved_scope,  # type: ignore[arg-type]
        source_location=SourceLocation(xpath=_xpath(tree, element)),
    )
    section.blocks = _make_blocks(tree, element, document_id, section_id, counter)
    child_elements = _children(element, "section")
    section.children = [
        _parse_section(
            tree,
            child,
            document_id,
            f"{section_id}.{index}",
            section.path,
            counter,
            resolved_scope,
        )
        for index, child in enumerate(child_elements, start=1)
    ]
    return section


def _reference_definition(element: etree._Element) -> ReferenceDefinition | None:
    tag = element.get("anchor")
    if not tag:
        return None
    front = _first_child(element, "front")
    title_element = _first_child(front, "title") if front is not None else None
    title = _element_text(title_element) if title_element is not None else None
    document_id = None
    if front is not None:
        for series_info in _children(front, "seriesInfo"):
            if (series_info.get("name") or "").casefold() == "rfc":
                value = series_info.get("value")
                if value:
                    document_id = f"RFC{value}"
                    break
    return ReferenceDefinition(
        tag=tag,
        title=title,
        document_id=document_id,
        target_url=element.get("target"),
    )


def _external_entity_urls(tree: etree._ElementTree) -> dict[str, str | None]:
    dtd = tree.docinfo.internalDTD
    if dtd is None:
        return {}
    return {entity.name: entity.system_url for entity in dtd.iterentities()}


def _external_reference_definition(
    element: etree._Entity,
    entity_urls: dict[str, str | None],
) -> ReferenceDefinition | None:
    tag = element.name
    if not tag:
        return None

    document_id = None
    rfc_match = re.fullmatch(r"RFC0*(\d+)", tag, re.IGNORECASE)
    if rfc_match:
        document_id = f"RFC{int(rfc_match.group(1))}"
    elif tag.startswith("I-D."):
        document_id = f"draft-{tag.removeprefix('I-D.')}"

    return ReferenceDefinition(
        tag=tag,
        document_id=document_id,
        target_url=entity_urls.get(tag),
    )


def _body_section_scope(title: str, default: str = "body") -> str:
    if default != "body":
        return default
    normalized = title.casefold().strip()
    if normalized in {
        "acknowledgements",
        "acknowledgments",
        "author's address",
        "authors' addresses",
        "authors’ addresses",
        "contributors",
        "index",
    }:
        return "back"
    return default


def parse_xml_document(path: Path) -> Document:
    """Parse an XML2RFC source file into the canonical section/block tree."""
    parser = etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        remove_comments=True,
        recover=False,
        huge_tree=False,
    )
    try:
        tree = etree.parse(str(path), parser)
    except etree.XMLSyntaxError as exc:
        raise ValueError(f"Invalid XML in {path}: {exc}") from exc
    root = tree.getroot()
    if _local_name(root) != "rfc":
        raise ValueError(f"Expected an <rfc> root element in {path}")

    document_id = root.get("docName") or path.stem
    front = _first_child(root, "front")
    title_element = _first_child(front, "title") if front is not None else None
    title = _element_text(title_element) if title_element is not None else None
    counter = [0]
    sections: list[Section] = []

    if front is not None:
        abstract = _first_child(front, "abstract")
        if abstract is not None:
            abstract_section = Section(
                section_id="abstract",
                title="Abstract",
                path=["abstract"],
                analysis_scope="front",
                source_location=SourceLocation(xpath=_xpath(tree, abstract)),
            )
            abstract_section.blocks = _make_blocks(
                tree, abstract, document_id, abstract_section.section_id, counter
            )
            if abstract_section.blocks:
                sections.append(abstract_section)

    middle = _first_child(root, "middle")
    if middle is not None:
        for index, section_element in enumerate(_children(middle, "section"), start=1):
            sections.append(
                _parse_section(
                    tree,
                    section_element,
                    document_id,
                    str(index),
                    [],
                    counter,
                )
            )

    definitions: list[ReferenceDefinition] = []
    entity_urls = _external_entity_urls(tree)
    back = _first_child(root, "back")
    if back is not None:
        reference_groups = _children(back, "references")
        for group_index, group in enumerate(reference_groups, start=1):
            name_element = _first_child(group, "name")
            group_title = (
                _element_text(name_element)
                if name_element is not None
                else group.get("title") or "References"
            )
            section_id = "references" if len(reference_groups) == 1 else f"references.{group_index}"
            reference_section = Section(
                section_id=section_id,
                title=group_title,
                path=[section_id],
                analysis_scope="references",
                source_location=SourceLocation(xpath=_xpath(tree, group)),
            )
            for reference_element in group.iter():
                if isinstance(reference_element, etree._Entity):
                    definition = _external_reference_definition(reference_element, entity_urls)
                elif (
                    isinstance(reference_element.tag, str)
                    and _local_name(reference_element) in {"reference", "referencegroup"}
                ):
                    definition = _reference_definition(reference_element)
                else:
                    continue
                if definition is None:
                    continue
                definitions.append(definition)
                counter[0] += 1
                display = " — ".join(
                    value for value in (f"[{definition.tag}]", definition.title) if value
                )
                reference_section.blocks.append(
                    Block(
                        block_id=block_identifier(document_id, section_id, counter[0]),
                        index=counter[0],
                        block_type="reference",
                        text=display,
                        source_location=(
                            SourceLocation(
                                line_start=reference_element.sourceline,
                                line_end=reference_element.sourceline,
                            )
                            if isinstance(reference_element, etree._Entity)
                            else SourceLocation(xpath=_xpath(tree, reference_element))
                        ),
                    )
                )
            sections.append(reference_section)

        for appendix_index, section_element in enumerate(_children(back, "section"), start=1):
            appendix_id = _appendix_identifier(appendix_index)
            sections.append(
                _parse_section(
                    tree,
                    section_element,
                    document_id,
                    appendix_id,
                    [],
                    counter,
                    "appendix",
                )
            )

    warnings: list[str] = []
    if middle is None:
        warnings.append("XML document has no <middle> element")
    if not sections:
        warnings.append("No content sections were parsed")

    return Document(
        document_id=document_id,
        title=title,
        source_path=path.as_posix(),
        source_format="xml",
        document_type="internet-draft",
        source_sha256=file_sha256(path),
        sections=sections,
        reference_definitions=definitions,
        warnings=warnings,
    )


def _appendix_identifier(index: int) -> str:
    value = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        value = chr(ord("A") + remainder) + value
    return value
