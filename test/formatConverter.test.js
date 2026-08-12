"use strict";
const { test } = require("node:test");
const assert = require("node:assert");
const path = require("path");

const FormatConverter = require(path.join(__dirname, "..", "src/core/FormatConverter.js"));

const stubLogger = { debug: () => {}, error: () => {}, info: () => {}, warn: () => {} };

function makeConverter() {
    return new FormatConverter(stubLogger, {
        get config() {
            return { forceThinking: false, thinkingLevel: null, webSearch: false };
        },
    });
}

// ---- toolCallId -> name mapping (OpenAI Chat -> Google, translateOpenAIToGoogle) ----
test("translateOpenAIToGoogle maps tool_call_id via assistant tool_calls; missing -> unknown_function", async () => {
    const fc = makeConverter();
    const body = {
        messages: [
            { content: "weather?", role: "user" },
            {
                content: null,
                role: "assistant",
                tool_calls: [{ function: { arguments: "{}", name: "get_weather" }, id: "call_1", type: "function" }],
            },
            { content: "70", role: "tool", tool_call_id: "call_1" },
            { content: "x", role: "tool", tool_call_id: "missing_id" },
            { content: "y", name: "explicit_now", role: "tool", tool_call_id: "call_2" },
        ],
        model: "gpt-4o",
    };
    const { googleRequest } = await fc.translateOpenAIToGoogle(body);
    const fnParts = googleRequest.contents
        .filter(c => c.parts && c.parts.some(p => p.functionResponse))
        .flatMap(c => c.parts.filter(p => p.functionResponse));
    const names = fnParts.map(p => p.functionResponse.name);
    assert.deepStrictEqual(names, ["get_weather", "unknown_function", "explicit_now"]);
});

// ---- toolCallId -> name mapping (OpenAI Responses -> Google, translateOpenAIResponseToGoogle) ----
test("translateOpenAIResponseToGoogle maps call_id via function_call; missing -> unknown_function; adds thoughtSignature", async () => {
    const fc = makeConverter();
    const body = {
        input: [
            { content: [{ text: "weather?", type: "input_text" }], role: "user", type: "message" },
            { arguments: "{}", call_id: "fc_1", name: "get_weather", type: "function_call" },
            { call_id: "fc_1", output: "70", type: "function_call_output" },
            { call_id: "missing", output: "x", type: "function_call_output" },
            { call_id: "fc_2", name: "explicit_now", output: "y", type: "function_call_output" },
        ],
        model: "gpt-5",
    };
    const { googleRequest } = await fc.translateOpenAIResponseToGoogle(body);
    const modelParts = googleRequest.contents
        .filter(c => c.parts && c.parts.some(p => p.functionCall))
        .flatMap(c => c.parts.filter(p => p.functionCall));
    const fnCall = modelParts[0].functionCall;
    assert.strictEqual(fnCall.name, "get_weather");
    assert.strictEqual(modelParts[0].thoughtSignature, FormatConverter.DUMMY_THOUGHT_SIGNATURE);
    const fnParts = googleRequest.contents
        .filter(c => c.parts && c.parts.some(p => p.functionResponse))
        .flatMap(c => c.parts.filter(p => p.functionResponse));
    const names = fnParts.map(p => p.functionResponse.name);
    assert.deepStrictEqual(names, ["get_weather", "unknown_function", "explicit_now"]);
});

// ---- consecutive same-role merge (OpenAI Chat -> Google) ----
test("translateOpenAIToGoogle merges consecutive tool messages into one user message", async () => {
    const fc = makeConverter();
    const body = {
        messages: [
            {
                content: null,
                role: "assistant",
                tool_calls: [{ function: { arguments: "{}", name: "a" }, id: "c1", type: "function" }],
            },
            { content: "1", role: "tool", tool_call_id: "c1" },
            { content: "2", role: "tool", tool_call_id: "c1" },
        ],
        model: "gpt-4o",
    };
    const { googleRequest } = await fc.translateOpenAIToGoogle(body);
    const userContents = googleRequest.contents.filter(c => c.role === "user");
    assert.strictEqual(userContents.length, 1, "consecutive tool messages should merge into a single user content");
    const partCount = userContents[0].parts.filter(p => p.functionResponse).length;
    assert.strictEqual(partCount, 2);
});

test("translateOpenAIResponseToGoogle merges consecutive function_call_output into one user message", async () => {
    const fc = makeConverter();
    const body = {
        input: [
            { arguments: "{}", call_id: "fc_1", name: "get_weather", type: "function_call" },
            { call_id: "fc_1", output: "1", type: "function_call_output" },
            { call_id: "fc_1", output: "2", type: "function_call_output" },
        ],
        model: "gpt-5",
    };
    const { googleRequest } = await fc.translateOpenAIResponseToGoogle(body);
    const userContents = googleRequest.contents.filter(c => c.role === "user");
    assert.strictEqual(
        userContents.length,
        1,
        "consecutive function_call_output should merge into a single user content"
    );
    const partCount = userContents[0].parts.filter(p => p.functionResponse).length;
    assert.strictEqual(partCount, 2);
});

// ---- functionCall-with-thoughtSignature preserved (OpenAI Chat stream translator) ----
test("translateGoogleToOpenAIStream preserves a functionCall part that carries thoughtSignature", () => {
    const fc = makeConverter();
    const chunk = JSON.stringify({
        candidates: [
            {
                content: {
                    parts: [{ functionCall: { args: { city: "SF" }, name: "get_weather" }, thoughtSignature: "sig" }],
                },
                finishReason: "STOP",
            },
        ],
    });
    const out = fc.translateGoogleToOpenAIStream(chunk, "gemini-2.5-flash-lite", {});
    assert.ok(typeof out === "string", `expected string, got ${String(out)}`);
    assert.ok(out.includes("tool_calls"), out.slice(0, 200));
    assert.ok(out.includes("get_weather"), out.slice(0, 200));
});

// ---- functionCall-with-thoughtSignature preserved (OpenAI Chat non-stream converter) ----
test("convertGoogleToOpenAINonStream preserves a functionCall part that carries thoughtSignature", () => {
    const fc = makeConverter();
    const resp = {
        candidates: [
            {
                content: {
                    parts: [{ functionCall: { args: { city: "SF" }, name: "get_weather" }, thoughtSignature: "sig" }],
                },
                finishReason: "STOP",
            },
        ],
    };
    const out = fc.convertGoogleToOpenAINonStream(resp, "gemini-2.5-flash-lite");
    const toolCalls = out.choices[0].message.tool_calls;
    assert.ok(Array.isArray(toolCalls) && toolCalls.length === 1, JSON.stringify(out.choices[0].message));
    assert.strictEqual(toolCalls[0].function.name, "get_weather");
});
