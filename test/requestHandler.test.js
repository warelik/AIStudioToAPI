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
test("_isEmptyUpstreamResponse: terminal STOP with thought text part is NOT empty (thought parts count as content)", () => {
    const rh = makeHandler();
    const resp = {
        candidates: [{ content: { parts: [{ thought: true, text: "hmm" }] }, finishReason: "STOP" }],
        usageMetadata: { candidatesTokenCount: 0, thoughtsTokenCount: 5 },
    };
    assert.strictEqual(rh._isEmptyUpstreamResponse(resp), false);
});

test("_isEmptyUpstreamResponse: candidates:[] header frame is NOT empty (no terminal evidence)", () => {
    const rh = makeHandler();
    assert.strictEqual(rh._isEmptyUpstreamResponse({ candidates: [] }), false);
    assert.strictEqual(rh._isEmptyUpstreamResponse({ candidates: [], usageMetadata: { promptTokenCount: 100 } }), false);
});

test("_isEmptyUpstreamResponse: choices:[] usage-only frame is NOT empty", () => {
    const rh = makeHandler();
    assert.strictEqual(rh._isEmptyUpstreamResponse({ choices: [] }), false);
    assert.strictEqual(rh._isEmptyUpstreamResponse({ choices: [], usage: { prompt_tokens: 50 } }), false);
});

test("_isEmptyUpstreamResponse: blocked promptFeedback is NOT empty (pass through, no switch)", () => {
    const rh = makeHandler();
    const resp = {
        candidates: [],
        promptFeedback: { blockReason: "SAFETY" },
    };
    assert.strictEqual(rh._isEmptyUpstreamResponse(resp), false);
});
test("_isEmptyUpstreamResponse: terminal STOP thinking-only with thoughtsTokenCount but no content is NOT empty", () => {
    const rh = makeHandler();
    const resp = {
        candidates: [{ content: { parts: [{ thought: true }] }, finishReason: "STOP" }],
        usageMetadata: { candidatesTokenCount: 0, thoughtsTokenCount: 3 },
    };
    assert.strictEqual(rh._isEmptyUpstreamResponse(resp), false);
});

// ---- OpenAI Response API fake stream: terminal empty upstream routes to switch+retry ----
test("Response API fake stream: empty upstream body is judged and routed to switch+retry, not forwarded", async () => {
    const rh = makeHandler();
    rh.logger = stubLogger;
    rh.config = { streamingMode: "fake", switchOnUses: 0, thinkingLevel: null, forceThinking: false };
    rh.needsSwitchingAfterRequest = false;
    rh.timeouts = { FAKE_STREAM: 100 };

    let switched = false;
    let errorSent = false;
    let dumped = false;
    let translated = false;

    rh.authSwitcher = {
        incrementUsageCount: () => 0,
        handleRequestFailureAndSwitch: async () => { switched = true; },
    };

    // Empty upstream: the tail queue delivers a STREAM_END with no content data,
    // so the accumulated fullBody stays empty and must be judged terminal-empty.
    const fakeQueue = { dequeue: async () => ({ type: "STREAM_END" }) };

    rh.connectionRegistry = {
        createMessageQueue: () => fakeQueue,
        removeMessageQueue: () => {},
    };
    rh._generateRequestId = () => "test-fake-empty";
    rh._startTrackedRequest = () => {};
    rh._setResponseApiFormat = (res, fmt) => { res.__responseApiFormat = fmt; };
    rh._ensureBrowserBackedRequestReady = async () => true;
    rh._setupClientDisconnectHandler = () => {};
    rh._initializeProxyRequestAttempt = () => {};
    rh._updateTrackedRequest = () => {};
    rh._getUsageStatsService = () => null;
    rh._executeRequestWithRetries = async () => ({ success: true, queue: fakeQueue });
    rh._forwardRequest = async () => {};
    rh._dumpUpstreamCorrelation = () => { dumped = true; };
    rh._handleRequestError = () => { errorSent = true; };
    rh._finalizeTrackedRequest = () => {};
    rh._isResponseWritable = () => true;
    rh._handleQueueTimeout = () => {};

    // Must NOT be reached: an empty upstream must not translate into a client stream.
    rh.formatConverter.translateGoogleToResponseAPIStream = () => { translated = true; };
    // Translate the outgoing OpenAI Responses request into Gemini deterministically.
    rh.formatConverter.translateOpenAIResponseToGoogle = () => ({
        googleRequest: { contents: [{ role: "user", parts: [{ text: "hi" }] }] },
        cleanModelName: "gemini-2.5-flash",
        modelStreamingMode: null,
    });

    const res = {
        headersSent: false,
        writableEnded: false,
        __responseApiSeq: null,
        status: () => ({ set: () => {} }),
        write: () => true,
        end: () => { res.writableEnded = true; },
    };
    const req = {
        body: { stream: true, input: "hi", model: "gpt-4o-mini" },
        headers: {},
        method: "POST",
        url: "/v1/responses",
        protocol: "http",
    };

    // The fake-stream keep-alive timer (12-18s) is left pending after the request finishes and
    // would hold the test runner's event loop open. Replace long timers with an immediate no-op.
    const realSetTimeout = global.setTimeout;
    global.setTimeout = (fn, ms, ...args) =>
        ms >= 1000 ? realSetTimeout(() => {}, 0, ...args) : realSetTimeout(fn, ms, ...args);
    try {
        await rh.processOpenAIResponseRequest(req, res);
    } finally {
        global.setTimeout = realSetTimeout;
    }

    assert.strictEqual(switched, true, "empty upstream fake stream must route to account switch + retry");
    assert.strictEqual(errorSent, true, "empty upstream fake stream must send an error to the client");
    assert.strictEqual(dumped, true, "empty upstream fake stream must write a correlation dump");
    assert.strictEqual(translated, false, "empty upstream fake stream must not translate/send an empty stream to the client");
});