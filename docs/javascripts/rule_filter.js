// Filters for the rule-code tables (docs/rules/index.md), in the style of ruff's rules
// page. Each rule's level, versions and area come from the #rule-data JSON that
// `scripts/ngff_rules.py docs` writes into the page.
(function () {
  function field(label, control) {
    const wrap = document.createElement("label");
    wrap.className = "rule-filter__field";
    const text = document.createElement("span");
    text.className = "rule-filter__label";
    text.textContent = label;
    wrap.append(text, control);
    return wrap;
  }

  function select(options) {
    const el = document.createElement("select");
    for (const [value, text] of [["", "All"], ...options]) {
      el.add(new Option(text, value));
    }
    return el;
  }

  function setup() {
    const source = document.getElementById("rule-data");
    const tables = [...document.querySelectorAll(".rule-table")];
    if (!source || !tables.length) return;
    const data = JSON.parse(source.textContent);

    const rows = [];
    for (const table of tables) {
      for (const tr of table.querySelectorAll("tbody tr")) {
        const code = tr.cells[0].textContent.trim();
        rows.push({ tr, table, text: tr.textContent.toLowerCase(), ...data.rules[code] });
      }
    }

    const search = document.createElement("input");
    search.type = "search";
    search.placeholder = "Code, name, or summary";
    const version = select(data.versions.map((v) => [v, v]));
    const area = select(
      Object.entries(data.areas).map(([key, name]) => [key, `${name} (${key})`]),
    );
    const level = select(["MUST", "SHOULD", "MAY"].map((l) => [l, l]));
    const clear = document.createElement("button");
    clear.type = "button";
    clear.textContent = "Clear filters";
    const count = document.createElement("p");
    count.className = "rule-filter__count";

    const bar = document.createElement("div");
    bar.className = "rule-filter";
    const fields = document.createElement("div");
    fields.className = "rule-filter__fields";
    fields.append(
      field("Search rules", search),
      field("Spec version", version),
      field("Area", area),
      field("Level", level),
      clear,
    );
    bar.append(fields, count);
    // (above the first section heading, so it reads as filtering every table)
    const first = tables[0].previousElementSibling;
    (first && /^H[1-6]$/.test(first.tagName) ? first : tables[0]).before(bar);

    function apply() {
      const words = search.value.toLowerCase().split(/\s+/).filter(Boolean);
      let shown = 0;
      const visible = new Map(tables.map((t) => [t, 0]));
      for (const row of rows) {
        const ok =
          words.every((w) => row.text.includes(w)) &&
          // (the yaozarrs checks have no versions: they apply to every version)
          (!version.value || !row.versions.length || row.versions.includes(version.value)) &&
          (!area.value || row.area === area.value) &&
          (!level.value || row.level === level.value);
        row.tr.hidden = !ok;
        shown += ok;
        if (ok) visible.set(row.table, visible.get(row.table) + 1);
      }
      // hide a section (its heading and table) when none of its rules match
      for (const [table, n] of visible) {
        table.hidden = !n;
        const heading = table.previousElementSibling;
        if (heading && /^H[1-6]$/.test(heading.tagName)) heading.hidden = !n;
      }
      const filtered = shown !== rows.length;
      count.textContent = filtered ? `${shown} of ${rows.length} rules` : `${rows.length} rules`;
      clear.disabled = !filtered;
    }
    for (const el of [search, version, area, level]) {
      el.addEventListener(el === search ? "input" : "change", apply);
    }
    clear.addEventListener("click", () => {
      search.value = version.value = area.value = level.value = "";
      apply();
    });
    apply();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setup);
  } else {
    setup();
  }
})();
