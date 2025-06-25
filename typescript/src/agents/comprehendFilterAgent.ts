import { Agent, AgentOptions } from "./agent";
import { ChatHistory, ConversationMessage, ParticipantRole } from "../types";
import { Logger } from "../utils/logger";
import {
  ComprehendClient,
  DetectSentimentCommand,
  DetectPiiEntitiesCommand,
  DetectToxicContentCommand,
  DetectSentimentCommandOutput,
  DetectPiiEntitiesCommandOutput,
  DetectToxicContentCommandOutput,
  LanguageCode
} from "@aws-sdk/client-comprehend";

// Interface for toxic content labels returned by Comprehend
interface ToxicContent {
  Name: "GRAPHIC" | "HARASSMENT_OR_ABUSE" | "HATE_SPEECH" | "INSULT" | "PROFANITY" | "SEXUAL" | "VIOLENCE_OR_THREAT";
  Score: number;
}

// Interface for toxic labels result structure
interface ToxicLabels {
  Labels: ToxicContent[];
  Toxicity: number;
}

// Type definition for custom check functions
type CheckFunction = (input: string) => Promise<string | null>;

// Extended options for ComprehendContentFilterAgent
export interface ComprehendFilterAgentOptions extends AgentOptions {
    region?: string;
    enableSentimentCheck?: boolean;
    enablePiiCheck?: boolean;
    enableToxicityCheck?: boolean;
    sentimentThreshold?: number;
    toxicityThreshold?: number;
    allowPii?: boolean;
    languageCode?: LanguageCode;
}

/**
 * ComprehendContentFilterAgent class
 *
 * This agent uses Amazon Comprehend to analyze and filter content based on
 * sentiment, PII, and toxicity. It can be configured to enable/disable specific
 * checks and allows for the addition of custom checks.
 */
export class ComprehendFilterAgent extends Agent {
  private comprehendClient: ComprehendClient;
  private customChecks: CheckFunction[] = [];

  private enableSentimentCheck: boolean;
  private enablePiiCheck: boolean;
  private enableToxicityCheck: boolean;
  private sentimentThreshold: number;
  private toxicityThreshold: number;
  private allowPii: boolean;
  private languageCode: LanguageCode;

  /**
   * Constructor for ComprehendContentFilterAgent
   * @param options - Configuration options for the agent
   */
  constructor(options: ComprehendFilterAgentOptions) {
    super(options);

    this.comprehendClient = options.region
      ? new ComprehendClient({ region: options.region })
      : new ComprehendClient();

    // Set default configuration using fields from options
    this.enableSentimentCheck = options.enableSentimentCheck ?? true;
    this.enablePiiCheck = options.enablePiiCheck ?? true;
    this.enableToxicityCheck = options.enableToxicityCheck ?? true;
    this.sentimentThreshold = options.sentimentThreshold ?? 0.7;
    this.toxicityThreshold = options.toxicityThreshold ?? 0.7;
    this.allowPii = options.allowPii ?? false;
    this.languageCode = this.validateLanguageCode(options.languageCode) ?? 'en';

    // Ensure at least one check is enabled
    if (!this.enableSentimentCheck &&
        !this.enablePiiCheck &&
        !this.enableToxicityCheck) {
      this.enableToxicityCheck = true;
    }
  }

  /**
   * Processes a user request by sending it to the Amazon Bedrock agent for processing.
   * @param inputText - The user input as a string.
   * @param userId - The ID of the user sending the request.
   * @param sessionId - The ID of the session associated with the conversation.
   * @param chatHistory - An array of Message objects representing the conversation history.
   * @param additionalParams - Optional additional parameters as key-value pairs.
   * @returns A Promise that resolves to a Message object containing the agent's response.
   */
  /* eslint-disable @typescript-eslint/no-unused-vars */
  async processRequest(
    inputText: string,
    userId: string,
    sessionId: string,
    chatHistory: ChatHistory,
    additionalParams?: Record<string, string>
  ): Promise<ConversationMessage> {
    try {
      const issues: string[] = [];

      // Run all checks in parallel
      const [sentimentResult, piiResult, toxicityResult] = await Promise.all([
        this.enableSentimentCheck ? this.detectSentiment(inputText) : null,
        this.enablePiiCheck ? this.detectPiiEntities(inputText) : null,
        this.enableToxicityCheck ? this.detectToxicContent(inputText) : null
      ]);

      // Process results
      if (this.enableSentimentCheck && sentimentResult) {
        const sentimentIssue = this.checkSentiment(sentimentResult);
        if (sentimentIssue) issues.push(sentimentIssue);
      }

      if (this.enablePiiCheck && piiResult) {
        const piiIssue = this.checkPii(piiResult);
        if (piiIssue) issues.push(piiIssue);
      }

      if (this.enableToxicityCheck && toxicityResult) {
        const toxicityIssue = this.checkToxicity(toxicityResult);
        if (toxicityIssue) issues.push(toxicityIssue);
      }

      // Run custom checks
      for (const check of this.customChecks) {
        const customIssue = await check(inputText);
        if (customIssue) issues.push(customIssue);
      }

      if (issues.length > 0) {
        Logger.logger.warn(`Content filter issues detected: ${issues.join('; ')}`);
        return null;  // Return null to indicate content should not be processed further
      }

      // If no issues, return the original input as a ConversationMessage
      return {
        role: ParticipantRole.ASSISTANT,
        content: [{ text: inputText }]
      };

    } catch (error) {
      Logger.logger.error("Error in ComprehendContentFilterAgent:", error);
      throw error;
    }
  }

  /**
   * Add a custom check function to the agent
   * @param check - A function that takes a string input and returns a Promise<string | null>
   */
  addCustomCheck(check: CheckFunction) {
    this.customChecks.push(check);
  }

  /**
   * Check sentiment of the input text
   * @param result - Result from Comprehend's sentiment detection
   * @returns A string describing the issue if sentiment is negative, null otherwise
   */
  private checkSentiment(result: DetectSentimentCommandOutput): string | null {
    if (result.Sentiment === 'NEGATIVE' &&
        result.SentimentScore?.Negative > this.sentimentThreshold) {
      return `Negative sentiment detected (${result.SentimentScore.Negative.toFixed(2)})`;
    }
    return null;
  }

  /**
   * Check for PII in the input text
   * @param result - Result from Comprehend's PII detection
   * @returns A string describing the issue if PII is detected, null otherwise
   */
  private checkPii(result: DetectPiiEntitiesCommandOutput): string | null {
    if (!this.allowPii && result.Entities && result.Entities.length > 0) {
      return `PII detected: ${result.Entities.map(e => e.Type).join(', ')}`;
    }
    return null;
  }

  /**
   * Check for toxic content in the input text
   * @param result - Result from Comprehend's toxic content detection
   * @returns A string describing the issue if toxic content is detected, null otherwise
   */
  private checkToxicity(result: DetectToxicContentCommandOutput): string | null {
    const toxicLabels = this.getToxicLabels(result);
    if (toxicLabels.length > 0) {
      return `Toxic content detected: ${toxicLabels.join(', ')}`;
    }
    return null;
  }

  /**
   * Detect sentiment using Amazon Comprehend
   * @param text - Input text to analyze
   */
  private async detectSentiment(text: string) {
    const command = new DetectSentimentCommand({
      Text: text,
      LanguageCode: this.languageCode
    });
    return this.comprehendClient.send(command);
  }

  /**
   * Detect PII entities using Amazon Comprehend
   * @param text - Input text to analyze
   */
  private async detectPiiEntities(text: string) {
    const command = new DetectPiiEntitiesCommand({
      Text: text,
      LanguageCode: this.languageCode
    });
    return this.comprehendClient.send(command);
  }

  /**
   * Detect toxic content using Amazon Comprehend
   * @param text - Input text to analyze
   */
  private async detectToxicContent(text: string) {
    const command = new DetectToxicContentCommand({
      TextSegments: [{ Text: text }],
      LanguageCode: this.languageCode
    });
    return this.comprehendClient.send(command);
  }

  /**
   * Extract toxic labels from the Comprehend response
   * @param toxicityResult - Result from Comprehend's toxic content detection
   * @returns Array of toxic label names that exceed the threshold
   */
  private getToxicLabels(toxicityResult: DetectToxicContentCommandOutput): string[] {
    const toxicLabels: string[] = [];

    if (toxicityResult.ResultList && Array.isArray(toxicityResult.ResultList)) {
      toxicityResult.ResultList.forEach((result: ToxicLabels) => {
        if (result.Labels && Array.isArray(result.Labels)) {
          result.Labels.forEach((label: ToxicContent) => {
            if (label.Score > this.toxicityThreshold) {
              toxicLabels.push(label.Name);
            }
          });
        }
      });
    }

    return toxicLabels;
  }

  /**
   * Set the language code for Comprehend operations
   * @param languageCode - The ISO 639-1 language code
   */
  setLanguageCode(languageCode: LanguageCode): void {
    const validatedLanguageCode = this.validateLanguageCode(languageCode);
    if (validatedLanguageCode) {
      this.languageCode = validatedLanguageCode;
    } else {
      throw new Error(`Invalid language code: ${languageCode}`);
    }
  }

  /**
   * Validate the provided language code
   * @param languageCode - The language code to validate
   * @returns The validated LanguageCode or undefined if invalid
   */
  private validateLanguageCode(languageCode: LanguageCode | undefined): LanguageCode | undefined {
    if (!languageCode) return undefined;

    const validLanguageCodes: LanguageCode[] = [
      'en', 'es', 'fr', 'de', 'it', 'pt', 'ar', 'hi', 'ja', 'ko', 'zh', 'zh-TW'
    ];

    return validLanguageCodes.includes(languageCode) ? languageCode : undefined;
  }
}