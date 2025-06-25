import { OpenAIAgentOptionsWithAuth } from "../agents/openAIAgent";
import { Logger } from "./logger";
import OpenAI from "openai";

export class LLMUtils {
  private client: OpenAI;
  private model: string;
  private maxFieldSize: number;

  constructor(options: OpenAIAgentOptionsWithAuth) {
    if (!options.client) {
      throw new Error("LLM client is required");
    }
    if (!options.model) {
      throw new Error("LLM Model is required");
    }
    this.client = options.client;
    this.model = options.model;
    this.maxFieldSize = 3500; //less than the 4kb limit for dynamodb
  }

  // Generate summary using LLM with size management
  public async generateSummary(conversationText: string): Promise<string> {

    // Check if conversation text is too large for LLM context
    const maxContextSize = 100000; // Adjust based on your LLM's context limit
    let textToSummarize = conversationText;

    if (conversationText.length > maxContextSize) {
      Logger.logger.info(
        `Conversation too large for single summary, truncating...`
      );
      textToSummarize =
        conversationText.substring(0, maxContextSize) +
        "\n\n[... conversation truncated ...]";
    }

    // const prompt = `Please provide a concise but comprehensive summary of the following conversation. Focus on key decisions, important context, ongoing topics, and any critical information needed to continue the conversation effectively. Keep the summary under 3000 characters to ensure it fits within storage limits:`;

    const prompt = `Summarize the following conversation, highlighting key topics, decisions, and important context that would be useful for future reference:
    \n${conversationText}\n Provide a concise summary (2-3 paragraphs) focusing on: \n1. Main topics discussed \n2. Key decisions or conclusion \n3. Important context for future conversations`;


    try {
      if (!this.client) {
        throw new Error("LLM client not configured");
      }

      const messages = [
        { role: "system", content: prompt },
        { role: "user" as const, content: textToSummarize },
      ] as OpenAI.Chat.ChatCompletionMessageParam[];

      // Use your LLM client here
      const chatCompletion = await this.client.chat.completions.create({
        messages: messages,
        model: this.model,
      });

      let summary =
        chatCompletion.choices[0]?.message?.content ||
        "Summary generation failed";

      // Ensure summary doesn't exceed our size limit
      if (Buffer.byteLength(summary, "utf8") > this.maxFieldSize) {
        Logger.logger.info("Generated summary too large, truncating...");
        // Truncate to fit within limit, ensuring we don't break mid-word
        const maxBytes = this.maxFieldSize - 100; // Buffer for safety
        let truncated = summary;
        while (Buffer.byteLength(truncated, "utf8") > maxBytes) {
          const words = truncated.split(" ");
          words.pop();
          truncated = words.join(" ") + "...";
        }
        summary = truncated;
      }

      return summary;
    } catch (error) {
      Logger.logger.error("Error generating summary:", error);
      // Fallback summary
      const fallbackSummary = `Conversation summary generation failed. You dont have access to summary of old chats and only have access to the last 10 messages.Apologies.`;
      return fallbackSummary;
    }
  }

  // Split large content into chunks
    splitContent(content: string): string[] {
    const chunks: string[] = [];
    let currentChunk = "";
    const words = content.split(" ");

    for (const word of words) {
      const testChunk = currentChunk + (currentChunk ? " " : "") + word;
      if (Buffer.byteLength(testChunk, "utf8") > this.maxFieldSize) {
        if (currentChunk) {
          chunks.push(currentChunk);
          currentChunk = word;
        } else {
          // Single word is too large, split by characters
          chunks.push(word.substring(0, this.maxFieldSize));
          currentChunk = word.substring(this.maxFieldSize);
        }
      } else {
        currentChunk = testChunk;
      }
    }

    if (currentChunk) {
      chunks.push(currentChunk);
    }

    return chunks;
  }
}