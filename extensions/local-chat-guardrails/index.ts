import type { OpenClawPluginApi } from "openclaw/plugin-sdk";

const LOCAL_CHAT_CHANNEL = "webchat";

const LOCAL_CHAT_GUIDANCE = [
  "## Local Chat Guardrails",
  "- When the active conversation is local webchat/CLI, answer in plain assistant text in the current chat.",
  "- Do not begin the reply with routing tags like [[reply_to_current]] or [[reply_to:<id>]].",
  "- Do not use the `message` tool for the user's inline reply; answer directly instead.",
  "- If you need clarification, ask for it in normal text in the same conversation.",
].join("\n");

function isLocalInteractiveRun(ctx: {
  messageProvider?: string;
  channelId?: string;
}): boolean {
  return ctx.messageProvider === LOCAL_CHAT_CHANNEL || ctx.channelId === LOCAL_CHAT_CHANNEL;
}

export default function register(api: OpenClawPluginApi) {
  api.on("before_prompt_build", (_event, ctx) => {
    if (!isLocalInteractiveRun(ctx)) {
      return {};
    }
    return {
      appendSystemContext: LOCAL_CHAT_GUIDANCE,
    };
  });

  api.logger.info("local-chat-guardrails: activated");
}
