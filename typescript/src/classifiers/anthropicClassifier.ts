import { error } from "console";
import {
  ANTHROPIC_MODEL_ID_CLAUDE_3_5_SONNET,
  ConversationMessage,
  ParticipantRole,
} from "../types";
import { isClassifierToolInput } from "../utils/helpers";
import { Logger } from "../utils/logger";
import { Classifier, ClassifierResult } from "./classifier";
import { Anthropic } from "@anthropic-ai/sdk";

export interface AnthropicClassifierOptions {
  // Optional: The ID of the Anthropic model to use for classification
  // If not provided, a default model may be used
  modelId?: string;

  logRequest?: boolean;

  // Optional: Configuration for the inference process
  inferenceConfig?: {
    // Maximum number of tokens to generate in the response
    maxTokens?: number;

    // Controls randomness in output generation
    // Higher values (e.g., 0.8) make output more random, lower values (e.g., 0.2) make it more deterministic
    temperature?: number;

    // Controls diversity of output via nucleus sampling
    // 1.0 considers all tokens, lower values (e.g., 0.9) consider only the most probable tokens
    topP?: number;

    // Array of sequences that will stop the model from generating further tokens when encountered
    stopSequences?: string[];
  };

  // The API key for authenticating with Anthropic's services
  apiKey: string;
}

export class AnthropicClassifier extends Classifier {
  private client: Anthropic;
  protected inferenceConfig: {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    stopSequences?: string[];
  };

  private tools: Anthropic.Tool[] = [
    {
      name: 'analyzePrompt',
      description: 'Analyze the user input and provide structured output',
      input_schema: {
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
  ];


  constructor(options: AnthropicClassifierOptions) {
    super();

    if (!options.apiKey) {
      throw new Error("Anthropic API key is required");
    }
    this.client = new Anthropic({ apiKey: options.apiKey });
    this.logRequest = options.logRequest ?? false;
    this.modelId = options.modelId || ANTHROPIC_MODEL_ID_CLAUDE_3_5_SONNET;
    // Set default value for max_tokens if not provided
    const defaultMaxTokens = 4096; // You can adjust this default value as needed
    this.inferenceConfig = {
      maxTokens: options.inferenceConfig?.maxTokens ?? defaultMaxTokens,
      temperature: options.inferenceConfig?.temperature,
      topP: options.inferenceConfig?.topP,
      stopSequences: options.inferenceConfig?.stopSequences,
    };

}

/* eslint-disable @typescript-eslint/no-unused-vars */
async processRequest(
    inputText: string,
    chatHistory: ConversationMessage[]
  ): Promise<ClassifierResult> {
    const userMessage: Anthropic.MessageParam = {
      role: ParticipantRole.USER,
      content: inputText,
    };
 
    let retry = true;
    let executionCount = 0;
    while(retry){
      retry = false;
      executionCount = executionCount+1;
      try {
        const req = {
          model: this.modelId,
          max_tokens: this.inferenceConfig.maxTokens,
          messages: [userMessage],
          system: this.systemPrompt,
          temperature: this.inferenceConfig.temperature,
          top_p: this.inferenceConfig.topP,
          tools: this.tools
        };
        const response = await this.client.messages.create(req);
        
  
        if(this.logRequest){
          console.log("\n\n---- Classifier ----");
          console.log(JSON.stringify(req));
          console.log(JSON.stringify(response));
          console.log("\n\n");
        }
  
        const modelStats = [];
        const obj = {};
        obj["id"] = response.id;
        obj["model"] = response.model;
        obj["usage"] = response.usage;
        obj["from"] = "anthropic_classifier";
        modelStats.push(obj);
        Logger.logger.info(`Anthropic Classifier Usage: `, JSON.stringify(obj));
        const toolUse = response.content.find(
          (content): content is Anthropic.ToolUseBlock => content.type === "tool_use"
        );
  
        if (!toolUse) {
          throw new Error("Classifier Error: No tool use found in the response");
        }
  
        if (!isClassifierToolInput(toolUse.input)) {
          throw new Error("Classifier Error: Tool input does not match expected structure");
        }
  
  
        // Create and return IntentClassifierResult
        const intentClassifierResult: ClassifierResult = {
          selectedAgent: this.getAgentById(toolUse.input.selected_agent),
          confidence: parseFloat(toolUse.input.confidence),
          modelStats: modelStats
        };
        return intentClassifierResult;
  
      } catch (error) {
        Logger.logger.error("Classifier Error: Error classifying request:", error);

        if(error.error.type === "overloaded_error"){
          if(executionCount < 3){
            retry = true;
            await delay(executionCount*500);
            Logger.logger.info(`Classifier Error: Overload Error: retry: ${executionCount}  delay ${executionCount*500}ms `);
          }else{
            Logger.logger.info(`Classifier Error: Exceeded retry count for overload error`);
            throw error;
          }
        }else{
          // Instead of returning a default result, we'll throw the error
          throw error;
        }
        
      }
    }
    throw error("Classifier Error: Please try again.");
  }


}

function delay(t: number) {
  return new Promise(resolve => {
    setTimeout(resolve, t);
  });
}
