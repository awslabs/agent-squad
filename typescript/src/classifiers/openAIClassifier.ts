import OpenAI from "openai";
import {
  ConversationMessage,
  OPENAI_MODEL_ID_GPT_O_MINI,
} from "../types";
import { isClassifierToolInput } from "../utils/helpers";
import { Logger } from "../utils/logger";
import { Classifier, ClassifierResult } from "./classifier";
import { ChatCompletionCreateParamsNonStreaming } from "openai/resources";

export interface OpenAIClassifierOptions {
  // Optional: The ID of the OpenAI model to use for classification
  // If not provided, a default model may be used
  modelId?: string;

  logRequest?: boolean;

  // Optional: Configuration for the inference process
  inferenceConfig?: {
    // Maximum number of tokens to generate in the response
    maxTokens?: number;

    // Controls randomness in output generation
    temperature?: number;

    // Controls diversity of output via nucleus sampling
    topP?: number;

    // Array of sequences that will stop the model from generating further tokens
    stopSequences?: string[];
  };

  // The API key for authenticating with OpenAI's services
  apiKey: string;
}

export class OpenAIClassifier extends Classifier {
  private client: OpenAI;
  protected inferenceConfig: {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    stopSequences?: string[];
  };

  private tools: OpenAI.ChatCompletionTool[] = [
    {
      type: "function",
      function: {
        name: 'analyzePrompt',
        description: 'Analyze the user input and provide structured output',
        parameters: {
          type: 'object',
          properties: {
            userinput: {
              type: 'string',
              description: 'The original user input',
            },
            selected_agent: {
              type: 'string',
              description: 'The name of the selected agent',
            },
            confidence: {
              type: 'number',
              description: 'Confidence level between 0 and 1',
            },
          },
          required: ['userinput', 'selected_agent', 'confidence'],
        },
      },
    },
  ];

  constructor(options: OpenAIClassifierOptions) {
    super();

    if (!options.apiKey) {
      throw new Error("OpenAI API key is required");
    }
    this.client = new OpenAI({ apiKey: options.apiKey });
    this.logRequest = options.logRequest ?? false;
    this.modelId = options.modelId || OPENAI_MODEL_ID_GPT_O_MINI;

    const defaultMaxTokens = 4096;
    this.inferenceConfig = {
      maxTokens: options.inferenceConfig?.maxTokens ?? defaultMaxTokens,
      temperature: options.inferenceConfig?.temperature,
      topP: options.inferenceConfig?.topP,
      stopSequences: options.inferenceConfig?.stopSequences,
    };
  }

  /**
   * Method to process a request.
   * This method must be implemented by all concrete agent classes.
   *
   * @param inputText - The user input as a string.
   * @param chatHistory - An array of Message objects representing the conversation history.
   * @param additionalParams - Optional additional parameters as key-value pairs.
   * @returns A Promise that resolves to a Message object containing the agent's response.
   */
  /* eslint-disable @typescript-eslint/no-unused-vars */
  async processRequest(
    inputText: string,
    chatHistory: ConversationMessage[]
  ): Promise<ClassifierResult> {
    const messages: OpenAI.ChatCompletionMessageParam[] = [
      {
        role: 'system',
        content: this.systemPrompt
      },
      {
        role: 'user',
        content: inputText
      }
    ];

    try {

      const req: ChatCompletionCreateParamsNonStreaming = {
        model: this.modelId,
        messages: messages,
        max_tokens: this.inferenceConfig.maxTokens,
        temperature: this.inferenceConfig.temperature,
        top_p: this.inferenceConfig.topP,
        tools: this.tools,
        tool_choice: { type: "function", function: { name: "analyzePrompt" } }
      }

      const response = await this.client.chat.completions.create(req);
      
      if(this.logRequest){
        console.log("\n\n---- OpenAI Classifier ----");
        console.log(JSON.stringify(req));
        console.log(JSON.stringify(response));
        console.log("\n\n");
      }

      const modelStats = [];
      const obj = {};
      obj["id"] = response.id;
      obj["model"] = response.model;
      obj["usage"] = response.usage;
      obj["from"] = "openai_classifier";
      modelStats.push(obj);
      Logger.logger.info(`OpenAI Classifier Usage: `, JSON.stringify(obj));

      const toolCall = response.choices[0]?.message?.tool_calls?.[0];

      if (!toolCall || toolCall.function.name !== "analyzePrompt") {
        throw new Error("No valid tool call found in the response");
      }

      const toolInput = JSON.parse(toolCall.function.arguments);

      if (!isClassifierToolInput(toolInput)) {
        throw new Error("Tool input does not match expected structure");
      }

      const intentClassifierResult: ClassifierResult = {
        selectedAgent: this.getAgentById(toolInput.selected_agent),
        confidence: parseFloat(toolInput.confidence),
        modelStats: modelStats
      };
      return intentClassifierResult;

    } catch (error) {
      Logger.logger.error("OpenAI Classfier Error processing request:", error);
      throw error;
    }
  }
}