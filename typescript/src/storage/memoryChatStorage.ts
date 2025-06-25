import { ChatStorage } from "./chatStorage";
import { ChatHistory, ConversationMessage, ParticipantRole, TimestampedMessage } from "../types";
import { Logger } from "../utils/logger";

export class InMemoryChatStorage extends ChatStorage {
  private conversations: Map<string, TimestampedMessage[]>;

  constructor() {
    super();
    this.conversations = new Map();
  }

  async saveChatMessage(
    userId: string,
    sessionId: string,
    agentId: string,
    newMessage: ConversationMessage,
    maxHistorySize?: number
  ): Promise<ConversationMessage[]> {
    const key = this.generateKey(userId, sessionId, agentId);
    let conversation = this.conversations.get(key) || [];

    if (super.isConsecutiveMessage(conversation, newMessage)) {
      Logger.logger.log(`> Consecutive ${newMessage.role} message detected for agent ${agentId}. Not saving.`);
      return this.removeTimestamps(conversation);
    }

    const timestampedMessage: TimestampedMessage = { ...newMessage, timestamp: Date.now() };
    conversation = [...conversation, timestampedMessage];
    conversation = super.trimConversation(conversation, maxHistorySize) as TimestampedMessage[];

    this.conversations.set(key, conversation);
    return this.removeTimestamps(conversation);
  }

  async fetchChat(
    userId: string,
    sessionId: string,
    agentId: string,
    maxHistorySize?: number
  ): Promise<ConversationMessage[]> {
    const key = this.generateKey(userId, sessionId, agentId);
    let conversation = this.conversations.get(key) || [];
    if (maxHistorySize !== undefined) {
      conversation = super.trimConversation(conversation, maxHistorySize) as TimestampedMessage[];
    }
    return this.removeTimestamps(conversation);
  }

  async fetchAllChats(
    userId: string,
    sessionId: string
  ): Promise<ChatHistory>  {
    const allMessages: TimestampedMessage[] = [];
    for (const [key, messages] of this.conversations.entries()) {
      const [storedUserId, storedSessionId, agentId] = key.split('#');
      if (storedUserId === userId && storedSessionId === sessionId) {
        // Add messages with their associated agentId
        allMessages.push(...messages.map(message => ({
          ...message,
          content: message.role === ParticipantRole.ASSISTANT 
            ? [{ text: `[${agentId}] ${message.content?.[0]?.text || ''}` }]
            : message.content
        })));
      }
    }
    // Sort messages by timestamp
    allMessages.sort((a, b) => a.timestamp - b.timestamp);
    const messages =  this.removeTimestamps(allMessages);
    return {messages}
  }

  private generateKey(userId: string, sessionId: string, agentId: string): string {
    return `${userId}#${sessionId}#${agentId}`;
  }

  private removeTimestamps(messages: TimestampedMessage[]): ConversationMessage[] {
    return messages.map(({ timestamp: _timestamp, ...message }) => message);
  }
}