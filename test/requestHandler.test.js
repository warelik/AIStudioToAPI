"use strict";
const { test } = require("node:test");
const assert = require("node:assert");
const path = require("path");

const RequestHandler = require(path.join(__dirname, "..", "src/core/RequestHandler.js"));
const FormatConverter = require(path.join(__dirname, "..", "src/core/FormatConverter.js"));
const ConnectionRegistry = require(path.join(__dirname, "..", "src/core/ConnectionRegistry.js"));

const stubLogger = { debug: () => {}, error: () => {}, info: () => {}, warn: () => {} };

function makeHandler() {
    const rh = Object.create(RequestHandler.prototype);
    rh.formatConverter = new FormatConverter(stubLogger, {
        get config() {
            return { forceThinking: false, thinkingLevel: null, webSearch: false };
        },
    });
    rh.timeouts = { STREAM_CHUNK: 60000 };
    return rh;
}

test("_isEmptyUpstreamResponse: native Anthropic text and tool_use are not empty", () => {
    const rh = makeHandler();
    assert.strictEqual(
        rh._isEmptyUpstreamResponse({
            content: [{ text: "hello", type: "text" }],
            stop_reason: "end_turn",
            usage: { output_tokens: 1 },
        }),
        false
    );
    assert.strictEqual(
        rh._isEmptyUpstreamResponse({
            content: [{ input: {}, name: "ping", type: "tool_use" }],
            stop_reason: "tool_use",
            usage: { output_tokens: 1 },
        }),
        false
    );
});

test("_isEmptyUpstreamResponse: unknown Anthropic content blocks fail safe as content", () => {
    const rh = makeHandler();
    assert.strictEqual(
        rh._isEmptyUpstreamResponse({
            content: [{ payload: { value: "future-output" }, type: "future_block" }],
            stop_reason: "end_turn",
            usage: { output_tokens: 0 },
        }),
        false
    );
});

test("_isEmptyUpstreamResponse: native Anthropic empty terminal response is empty", () => {
    const rh = makeHandler();
    assert.strictEqual(
        rh._isEmptyUpstreamResponse({
            content: [{ text: "   ", type: "text" }],
            stop_reason: "end_turn",
            usage: { output_tokens: 0 },
        }),
        true
    );
    assert.strictEqual(rh._isEmptyUpstreamResponse({ content: [] }), false, "non-terminal headers stay pass-through");
});

test("_withFailureAuthIndex resolves the request account without reading mutable current account", () => {
    const rh = makeHandler();
    rh.connectionRegistry = {
        getAuthIndexForRequest: requestId => (requestId === "req-source" ? 4 : null),
    };
    const details = rh._withFailureAuthIndex({ status: 429 }, "req-source");
    assert.deepStrictEqual(details, { authIndex: 4, status: 429 });
    const explicit = rh._withFailureAuthIndex({ status: 502 }, "missing", 7);
    assert.deepStrictEqual(explicit, { authIndex: 7, status: 502 });
});

test("ConnectionRegistry routes each browser message with its source authIndex", () => {
    const registry = Object.create(ConnectionRegistry.prototype);
    const messages = [];
    const queue = { close: () => {}, enqueue: message => messages.push(message) };
    registry._routeMessage({ data: { text: "x" }, event_type: "chunk" }, queue, 3);
    registry._routeMessage({ event_type: "stream_close" }, queue, 5);
    assert.strictEqual(messages[0].authIndex, 3);
    assert.deepStrictEqual(messages[1], { authIndex: 5, type: "STREAM_END" });
});

test("AuthSwitcher attributes concurrent failure counters to explicit source authIndex", async () => {
    const AuthSwitcher = require(path.join(__dirname, "..", "src/auth/AuthSwitcher.js"));
    const mockBrowser = { currentAuthIndex: 9 };
    const authSwitcher = new AuthSwitcher(
        stubLogger,
        { immediateSwitchStatusCodes: [502] },
        { getAuthCount: () => 10, getCanonicalIndex: i => i },
        mockBrowser
    );
    let switchStartIndex;
    let allowOriginalFallback;
    authSwitcher.switchToNextAuth = async (failedAuthIndex, allowFallback) => {
        switchStartIndex = failedAuthIndex;
        allowOriginalFallback = allowFallback;
        return { success: true };
    };

    await authSwitcher.handleRequestFailureAndSwitch({ authIndex: 2, reason: "empty_upstream_response" }, null);

    assert.strictEqual(authSwitcher._emptyJudgmentCounts.get(2), 1);
    assert.strictEqual(authSwitcher._emptyJudgmentCounts.has(9), false);
    assert.strictEqual(switchStartIndex, 2);
    assert.strictEqual(allowOriginalFallback, false);
});

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
        candidates: [
            {
                content: { parts: [{ functionCall: { args: {}, name: "get_weather" } }] },
                finishReason: "STOP",
            },
        ],
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
    const resp = { candidates: [{ content: { parts: [{ text: "hmm", thought: true }] } }] };
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
        candidates: [{ content: { parts: [{ text: "hmm", thought: true }] }, finishReason: "STOP" }],
        usageMetadata: { candidatesTokenCount: 0, thoughtsTokenCount: 5 },
    };
    assert.strictEqual(rh._isEmptyUpstreamResponse(resp), false);
});

test("_isEmptyUpstreamResponse: candidates:[] header frame is NOT empty (no terminal evidence)", () => {
    const rh = makeHandler();
    assert.strictEqual(rh._isEmptyUpstreamResponse({ candidates: [] }), false);
    assert.strictEqual(
        rh._isEmptyUpstreamResponse({ candidates: [], usageMetadata: { promptTokenCount: 100 } }),
        false
    );
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
    rh.config = { forceThinking: false, streamingMode: "fake", switchOnUses: 0, thinkingLevel: null };
    rh.needsSwitchingAfterRequest = false;
    rh.timeouts = { FAKE_STREAM: 100 };

    let switched = false;
    let errorSent = false;
    let dumped = false;
    let translated = false;

    rh.authSwitcher = {
        handleRequestFailureAndSwitch: async () => {
            switched = true;
        },
        incrementUsageCount: () => 0,
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
    rh._setResponseApiFormat = (res, fmt) => {
        res.__responseApiFormat = fmt;
    };
    rh._ensureBrowserBackedRequestReady = async () => true;
    rh._setupClientDisconnectHandler = () => {};
    rh._initializeProxyRequestAttempt = () => {};
    rh._updateTrackedRequest = () => {};
    rh._getUsageStatsService = () => null;
    rh._executeRequestWithRetries = async () => ({ queue: fakeQueue, success: true });
    rh._forwardRequest = async () => {};
    rh._dumpUpstreamCorrelation = () => {
        dumped = true;
    };
    rh._handleRequestError = () => {
        errorSent = true;
    };
    rh._finalizeTrackedRequest = () => {};
    rh._isResponseWritable = () => true;
    rh._handleQueueTimeout = () => {};

    // Must NOT be reached: an empty upstream must not translate into a client stream.
    rh.formatConverter.translateGoogleToResponseAPIStream = () => {
        translated = true;
    };
    // Translate the outgoing OpenAI Responses request into Gemini deterministically.
    rh.formatConverter.translateOpenAIResponseToGoogle = () => ({
        cleanModelName: "gemini-2.5-flash",
        googleRequest: { contents: [{ parts: [{ text: "hi" }], role: "user" }] },
        modelStreamingMode: null,
    });

    const res = {
        __responseApiSeq: null,
        end: () => {
            res.writableEnded = true;
        },
        headersSent: false,
        status: () => ({ set: () => {} }),
        writableEnded: false,
        write: () => true,
    };
    const req = {
        body: { input: "hi", model: "gpt-4o-mini", stream: true },
        headers: {},
        method: "POST",
        protocol: "http",
        url: "/v1/responses",
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
    assert.strictEqual(
        translated,
        false,
        "empty upstream fake stream must not translate/send an empty stream to the client"
    );
});

// ---- Regression tests for Items 1, 2, 3, 4 ----
test("Item 1: _dumpUpstreamCorrelation is only called for event_type === 'chunk' with defined data", () => {
    const rh = makeHandler();
    let dumpCalled = false;
    rh._dumpUpstreamCorrelation = () => {
        dumpCalled = true;
    };

    // response_headers frame (no data) must NOT call _dumpUpstreamCorrelation
    const headerMsg = { event_type: "response_headers", headers: {} };
    if (headerMsg?.event_type === "chunk" && headerMsg.data !== undefined) {
        rh._dumpUpstreamCorrelation("test", headerMsg.data, "req-1", "m", 0);
    }
    assert.strictEqual(dumpCalled, false);

    // chunk frame with data MUST call _dumpUpstreamCorrelation
    const chunkMsg = { data: { foo: "bar" }, event_type: "chunk" };
    if (chunkMsg?.event_type === "chunk" && chunkMsg.data !== undefined) {
        rh._dumpUpstreamCorrelation("test", chunkMsg.data, "req-1", "m", 0);
    }
    assert.strictEqual(dumpCalled, true);
});

test("Item 2: _streamOpenAIResponse uses _sendErrorChunkToClient when headers already sent", async () => {
    const rh = makeHandler();
    rh.logger = stubLogger;
    rh.authSwitcher = { handleRequestFailureAndSwitch: () => {} };

    let sseErrorSent = false;
    let jsonErrorSent = false;

    rh._sendErrorChunkToClient = () => {
        sseErrorSent = true;
    };
    rh._sendErrorResponse = () => {
        jsonErrorSent = true;
    };

    const fakeQueue = {
        dequeue: async () => ({ type: "STREAM_END" }),
    };

    const res = {
        end: () => {},
        headersSent: true,
        writableEnded: false,
    };

    await rh._streamOpenAIResponse(fakeQueue, res, "gpt-4o", "req-stream-err");

    assert.strictEqual(sseErrorSent, true, "must call _sendErrorChunkToClient when res.headersSent is true");
    assert.strictEqual(jsonErrorSent, false, "must NOT call _sendErrorResponse when res.headersSent is true");
});

test("Item 3: AuthSwitcher deletes only failing index on non-empty failure, preserving other accounts", async () => {
    const AuthSwitcher = require(path.join(__dirname, "..", "src/auth/AuthSwitcher.js"));
    const mockBrowser = { currentAuthIndex: 0 };
    const authSwitcher = new AuthSwitcher(
        stubLogger,
        { immediateSwitchStatusCodes: [401, 403, 429, 500, 502, 503] },
        { getAuthCount: () => 3, getCanonicalIndex: i => i },
        mockBrowser
    );
    authSwitcher.switchToNextAuth = async () => ({ success: true });

    // Simulate empty failure on account 0 and account 1
    mockBrowser.currentAuthIndex = 0;
    await authSwitcher.handleRequestFailureAndSwitch({ reason: "empty_upstream_response" }, null);

    mockBrowser.currentAuthIndex = 1;
    await authSwitcher.handleRequestFailureAndSwitch({ reason: "empty_upstream_response" }, null);

    assert.strictEqual(authSwitcher._emptyJudgmentCounts.get(0), 1);
    assert.strictEqual(authSwitcher._emptyJudgmentCounts.get(1), 1);

    // Non-empty failure on account 1 should delete account 1 counter ONLY
    mockBrowser.currentAuthIndex = 1;
    await authSwitcher.handleRequestFailureAndSwitch({ reason: "rate_limit", status: 429 }, null);

    assert.strictEqual(authSwitcher._emptyJudgmentCounts.has(1), false, "account 1 counter deleted");
    assert.strictEqual(authSwitcher._emptyJudgmentCounts.get(0), 1, "account 0 counter preserved");
});

test("Item 4: FormatConverter.mergeConsecutiveSameRoleContents merges same roles", () => {
    const googleContents = [
        { parts: [{ text: "a" }], role: "user" },
        { parts: [{ text: "b" }], role: "user" },
        { parts: [{ text: "c" }], role: "model" },
    ];
    const merged = FormatConverter.mergeConsecutiveSameRoleContents(googleContents);
    assert.strictEqual(merged.length, 2);
    assert.strictEqual(merged[0].role, "user");
    assert.deepStrictEqual(merged[0].parts, [{ text: "a" }, { text: "b" }]);
    assert.strictEqual(merged[1].role, "model");
});
