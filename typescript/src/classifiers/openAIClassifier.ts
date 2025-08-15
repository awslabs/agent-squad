import OpenAI from "openai";
import { ChatHistory, OPENAI_MODEL_ID_GPT_O_MINI } from "../types";
import { isClassifierToolInput } from "../utils/helpers";
import { Logger } from "../utils/logger";
import { Classifier, ClassifierResult } from "./classifier";
import { ChatCompletionCreateParamsNonStreaming } from "openai/resources";
import {fetchDescription} from "../utils/s3Utils"

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

  // private tools: OpenAI.ChatCompletionTool[] = [
  //   {
  //     type: "function",
  //     function: {
  //       name: "analyzePrompt",
  //       description: "Analyze the user input and provide structured output",
  //       parameters: {
  //         type: "object",
  //         properties: {
  //           userinput: {
  //             type: "string",
  //             description: "The original user input",
  //           },
  //           selected_agent: {
  //             type: "string",
  //             description: "The name of the selected agent",
  //           },
  //           confidence: {
  //             type: "number",
  //             description: "Confidence level between 0 and 1",
  //           },
  //         },
  //         required: ["userinput", "selected_agent", "confidence"],
  //       },
  //     },
  //   },
  // ];

  private tools: OpenAI.ChatCompletionTool[] = [
    {
      type: "function",
      function: {
        name: "analyzePrompt",
        description: "Analyze the user input and provide structured output",
        parameters: {
          type: "object",
          properties: {
            agents: {
              type: "array",
              items: {
                type: "object",
                properties: {
                userinput: {
                  type: "string",
                  description: "The original user input",
                },
                selected_agent: {
                  type: "string",
                  description: "The name of the selected agent",
                },
                confidence: {
                  type: "number",
                  description: "Confidence level between 0 and 1",
                },
              },
              required: ["userinput", "selected_agent", "confidence"],
              }
            }
          },
          required: ["agents"],
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
    chatHistory: ChatHistory
  ): Promise<ClassifierResult[]> {
    const messages: OpenAI.ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: this.systemPrompt,
      },
      {
        role: "user",
        content: inputText,
      },
    ];

    try {
      const req: ChatCompletionCreateParamsNonStreaming = {
        model: this.modelId,
        messages: messages,
        max_tokens: this.inferenceConfig.maxTokens,
        temperature: this.inferenceConfig.temperature,
        top_p: this.inferenceConfig.topP,
        tools: this.tools,
        tool_choice: { type: "function", function: { name: "analyzePrompt" } },
      };

      if (this.logRequest) {
        console.log("\n\n---- OpenAI Classifier ----");
        console.log(JSON.stringify(req));
      }

      const response = await this.client.chat.completions.create(req);

      if (this.logRequest) {
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

      const classifiedResults = [];

      if(toolInput['agents']){
      //iterate over tool calls for multiple agents
        for(const agent of toolInput['agents']){

          if (!isClassifierToolInput(agent)) {
            console.log( " ?? ", agent);
            throw new Error("Tool input does not match expected structure.");
          }
          //update agent description from s3 by replacing the current description which is summary.
          const selectedAgent = this.getAgentById(agent['selected_agent']);
              //update description from s3 only if s3 details are provided.
          /**
           * Ideally all agents from db should have description in s3.
           * But if we create a custom agent class itself, we can handle the description part in the custom class
           * So in that case, there is no s3 details to fetch as custom prompt or very detailed description is in class itself
           * if that class also wants to store description in s3, set the variable.
           */
          if (
            selectedAgent &&
            selectedAgent.s3details &&
            selectedAgent.s3details.indexOf("##") > 0
          ) {
            Logger.logger.info(
              `For selected agent fetching info from s3: ${selectedAgent.s3details}`
            );
            const s3details = selectedAgent.s3details;
            const [S3Bucket, fileId] = s3details.split("##");
            const description = await fetchDescription(S3Bucket, fileId);
            selectedAgent.description = description;
            Logger.logger.info(
              `For selected agent updated description from s3 : ${selectedAgent.description}`
            );
          }
          const intentClassifierResult: ClassifierResult = {
            selectedAgent: selectedAgent,
            confidence: parseFloat(toolInput.confidence),
            modelStats: modelStats,
          };
          classifiedResults.push(intentClassifierResult)
        }
      }else{
        Logger.logger.error("OpenAI Classfier Error processing request:");
      }
     

      return classifiedResults;
    } catch (error) {
      Logger.logger.error("OpenAI Classfier Error processing request:", error);
      throw error;
    }
  }
}
