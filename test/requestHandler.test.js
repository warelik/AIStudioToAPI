"use strict";
const { test } = require("node:test");
const assert = require("node:assert");
const path = require("path");

const RequestHandler = require(path.join(__dirname, "..", "src/core/RequestHandler.js"));
const FormatConverter = require(path.join(__dirname, "..", "src/core/FormatConverter.js"));

const stubLogger = { debug: () => {}, info: () => {}, warn: () => {}, error: () => {} };

function makeHandler() {
    const rh = Object.create(RequestHandler.prototype);
    rh.formatConverter = new FormatConverter(stubLogger, {
        get config() { return { forceThinking: false, thinkingLevel: null, webSearch: false }; },
        config: { forceThinking: false, thinkingLevel: null, webSearch: false },
    });
    return rh;
}

// ---- _extractSseEvents ----
test("_extractSseEvents splits events on LF blank line", () => {
    const rh = makeHandler();
    const { complete, remainder } = rh._extractSseEvents("data: a\n\ndata: b\n\n");
    assert.deepStrictEqual(complete, ["data: a", "data: b"]);
    assert.strictEqual(remainder, "");
});

test("_extractSseEvents splits events on CRLF blank line", () => {
    const rh = makeHandler();
    const { complete, remainder } = rh._extractSseEvents("data: a\r\n\r\ndata: b\r\n\r\n");
    assert.deepStrictEqual(complete, ["data: a", "data: b"]);
    assert.strictEqual(remainder, "");
});

test("_extractSseEvents keeps a fragmented trailing event in remainder", () => {
    const rh = makeHandler();
    const { complete, remainder } = rh._extractSseEvents("data: a\n\ndata: partial");
    assert.deepStrictEqual(complete, ["data: a"]);
    assert.strictEqual(remainder, "data: partial");
});

test("_extractSseEvents returns multiple complete events from one chunk", () => {
    const rh = makeHandler();
    const { complete, remainder } = rh._extractSseEvents("data: a\n\ndata: b\n\ndata: c\n\n");
    assert.deepStrictEqual(complete, ["data: a", "data: b", "data: c"]);
    assert.strictEqual(remainder, "");
});

test("_extractSseEvents preserves a trailing CR in the remainder", () => {
    const rh = makeHandler();
    const { complete, remainder } = rh._extractSseEvents("data: a\n\ndata: b\r");
    assert.deepStrictEqual(complete, ["data: a"]);
    assert.strictEqual(remainder, "data: b\r");
});

// ---- _translateCompleteSseEvent ----
test("_translateCompleteSseEvent returns null for an empty event", () => {
    const rh = makeHandler();
    assert.strictEqual(rh._translateCompleteSseEvent("", "gemini-2.5-flash-lite"), null);
    assert.strictEqual(rh._translateCompleteSseEvent("  \n ", "gemini-2.5-flash-lite"), null);
});

test("_translateCompleteSseEvent skips an event with no data line", () => {
    const rh = makeHandler();
    assert.strictEqual(rh._translateCompleteSseEvent("event: ping\nid: 1", "gemini-2.5-flash-lite"), null);
});

test("_translateCompleteSseEvent passes through [DONE]", () => {
    const rh = makeHandler();
    const out = rh._translateCompleteSseEvent("data: [DONE]", "gemini-2.5-flash-lite");
    assert.ok(typeof out === "string" && out.includes("[DONE]"), `got ${String(out)}`);
});

test("_translateCompleteSseEvent skips a non-JSON data payload", () => {
    const rh = makeHandler();
    assert.strictEqual(rh._translateCompleteSseEvent("data: not-json", "gemini-2.5-flash-lite"), null);
});

test("_translateCompleteSseEvent translates a valid JSON event and trims trailing CR", () => {
    const rh = makeHandler();
    const payload = JSON.stringify({ candidates: [{ content: { parts: [{ text: "hello hi" }] } }] });
    const out = rh._translateCompleteSseEvent(`data: ${payload}\r`, "gemini-2.5-flash-lite");
    assert.ok(typeof out === "string" && out.includes("hello hi"), `got ${String(out)}`);
});

// ---- _isEmptyUpstreamResponse ----
test("_isEmptyUpstreamResponse: pure tool call is NOT empty", () => {
    const rh = makeHandler();
    const resp = {
        candidates: [{
            content: { parts: [{ functionCall: { name: "get_weather", args: {} } }] },
            finishReason: "STOP",
        }],
    };
    assert.strictEqual(rh._isEmptyUpstreamResponse(resp), false);
});

test("_isEmptyUpstreamResponse: text content is NOT empty", () => {
    const rh = makeHandler();
    const resp = { candidates: [{ content: { parts: [{ text: "hello" }] }, finishReason: "STOP" }] };
    assert.strictEqual(rh._isEmptyUpstreamResponse(resp), false);
});

test("_isEmptyUpstreamResponse: reasoning-only non-terminal chunk is NOT empty", () => {
    const rh = makeHandler();
    const resp = { candidates: [{ content: { parts: [{ thought: true, text: "hmm" }] } }] };
    assert.strictEqual(rh._isEmptyUpstreamResponse(resp), false);
});

test("_isEmptyUpstreamResponse: whitespace-only + STOP + zero completion tokens is EMPTY", () => {
    const rh = makeHandler();
    const resp = {
        candidates: [{ content: { parts: [{ text: "   " }] }, finishReason: "STOP" }],
        usageMetadata: { candidatesTokenCount: 0, thoughtsTokenCount: 0 },
    };
    assert.strictEqual(rh._isEmptyUpstreamResponse(resp), true);
});

test("_isEmptyUpstreamResponse: empty string + STOP is EMPTY", () => {
    const rh = makeHandler();
    const resp = { candidates: [{ content: { parts: [] }, finishReason: "STOP" }] };
    assert.strictEqual(rh._isEmptyUpstreamResponse(resp), true);
});

test("_isEmptyUpstreamResponse: whitespace text WITH completion tokens is NOT empty", () => {
    const rh = makeHandler();
    const resp = {
        candidates: [{ content: { parts: [{ text: " " }] }, finishReason: "STOP" }],
        usageMetadata: { candidatesTokenCount: 3, thoughtsTokenCount: 0 },
    };
    assert.strictEqual(rh._isEmptyUpstreamResponse(resp), false);
});