// Browser (and Node) port of the notebook's lyric cleaning + TF-IDF + logistic regression.
// The model itself lives in model.json, written by export_model.py.
(function (root) {
  const SINGER_LABEL = /^[ \t]*[A-Z][A-Za-z'&.-]*(?:[ \t]+[A-Z][A-Za-z'&.-]*){0,2}[ \t]*:[ \t]*/gm;
  const BRACKET_TAG = /\[[^\]]*\]/g;
  const SECTION_LINE = /^[ \t]*(?:pre-?chorus|chorus|verse|bridge|intro|outro|hook|refrain|interlude|instrumental|solo)(?:[ \t]+(?:\d+x?|x\d+|\d+))?[ \t]*:?[ \t]*$/gm;

  // Same steps, same order as the notebook's cleaning cell.
  function clean(text) {
    let t = text.replace(/\\n/g, "\n").replace(/\\r/g, "\n").replace(/\\"/g, '"');
    t = t.replace(SINGER_LABEL, " ");              // PHANTOM:, Madame Giry:  (needs the capitals, so before lowercasing)
    t = t.toLowerCase();
    t = t.replace(BRACKET_TAG, " ");               // [Chorus], [Layzie Bone], [?]
    t = t.replace(SECTION_LINE, " ");              // bare "Chorus", "Instrumental", "Solo" lines
    t = t.replace(/[\r\n]+/g, " ");
    t = t.replace(/[‘’]/g, "'");
    t = t.normalize("NFKD").replace(/[̀-ͯ]/g, "");
    t = t.replace(/[^a-z]+/g, " ").replace(/\s+/g, " ").trim();
    return t;
  }

  function prepare(json) {
    const termIndex = new Map();
    json.terms.forEach((t, i) => termIndex.set(t, i));
    const coefOf = new Int32Array(json.terms.length).fill(-1);
    json.coef_index.forEach((termIdx, k) => { coefOf[termIdx] = k; });
    return {
      classes: json.classes,
      intercept: json.intercept,
      terms: json.terms,
      idf: json.idf,
      coef: Float64Array.from(json.coef),
      coefOf,
      termIndex,
      stop: new Set(json.stop),
      lemma: new Map(Object.entries(json.lemma)),
      sublinear: json.sublinear_tf,
    };
  }

  function tokens(text, m) {
    const out = [];
    for (const w of clean(text).split(" ")) {
      if (w.length < 2 || m.stop.has(w)) continue;
      const l = m.lemma.get(w) || w;
      if (!m.stop.has(l)) out.push(l);
    }
    return out;
  }

  // Returns null if no word of the lyrics is in the model's vocabulary.
  function predict(m, text) {
    const counts = new Map();
    for (const tok of tokens(text, m)) {
      const i = m.termIndex.get(tok);
      if (i !== undefined) counts.set(i, (counts.get(i) || 0) + 1);
    }
    if (counts.size === 0) return null;

    const entries = [];
    let norm = 0;
    for (const [i, c] of counts) {
      const v = (m.sublinear ? 1 + Math.log(c) : c) * m.idf[i];
      entries.push([i, v]);
      norm += v * v;
    }
    norm = Math.sqrt(norm);                         // L2 normalisation over every in-vocabulary term

    const C = m.classes.length;
    const logits = Array.from(m.intercept);
    for (const [i, v] of entries) {
      const k = m.coefOf[i];
      if (k < 0) continue;
      const x = v / norm;
      for (let c = 0; c < C; c++) logits[c] += x * m.coef[k * C + c];
    }
    const mx = Math.max(...logits);
    const exps = logits.map((z) => Math.exp(z - mx));
    const sum = exps.reduce((a, b) => a + b, 0);
    const ranked = m.classes
      .map((genre, c) => ({ genre, c, p: exps[c] / sum, logit: logits[c] }))
      .sort((a, b) => b.p - a.p);

    // Per-word contribution to one genre's score: tf-idf value x that genre's coefficient.
    function topWords(c, n = 6) {
      const words = [];
      for (const [i, v] of entries) {
        const k = m.coefOf[i];
        if (k >= 0) words.push({ word: m.terms[i], weight: (v / norm) * m.coef[k * C + c] });
      }
      return words.sort((a, b) => b.weight - a.weight).slice(0, n);
    }
    return { ranked, topWords, matchedTerms: counts.size };
  }

  const api = { clean, prepare, tokens, predict };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.GenrePredictor = api;
})(typeof self !== "undefined" ? self : this);
