import { Client, createClient } from '@libsql/client/web';
import { ConversationMessage, ParticipantRole } from "../types";
import { Logger } from "../utils/logger";
import { ChatStorage } from "./chatStorage";

export class SqlChatStorage extends ChatStorage {
  private client: Client;
  private initialized: Promise<void>;

  constructor(url: string, authToken?: string) {
    super();
    this.client = createClient({
      url,
      authToken,
    });
    this.initialized = this.initializeDatabase();
  }

  private async initializeDatabase() {
    try {
      // Create conversations table if it doesn't exist
      await this.client.execute(/*sql*/`
        CREATE TABLE IF NOT EXISTS conversations (
          user_id TEXT NOT NULL,
          session_id TEXT NOT NULL,
          agent_id TEXT NOT NULL,
          message_index INTEGER NOT NULL,
          role TEXT NOT NULL,
          content TEXT NOT NULL,
          timestamp INTEGER NOT NULL,
          PRIMARY KEY (user_id, session_id, agent_id, message_index)
        );
      `);

      // Create index for faster queries
      await this.client.execute(/*sql*/`
        CREATE INDEX IF NOT EXISTS idx_conversations_lookup
        ON conversations(user_id, session_id, agent_id);
      `);
    } catch (error) {
      Logger.logger.error("Error initializing database:", error);
      throw new Error("Database initialization error");
    }
  }

  async saveChatMessage(
    userId: string,
    sessionId: string,
    agentId: string,
    newMessage: ConversationMessage,
    maxHistorySize?: number
  ): Promise<ConversationMessage[]> {
    try {
      // Fetch existing conversation
      const existingConversation = await this.fetchChat(userId, sessionId, agentId);

      if (super.isConsecutiveMessage(existingConversation, newMessage)) {
        Logger.logger.log(`> Consecutive ${newMessage.role} message detected for agent ${agentId}. Not saving.`);
        return existingConversation;
      }

      // Begin transaction
      await this.client.transaction().then(async (tx) => {
        // Get the next message index
        const nextIndexResult = await tx.execute({
          sql: /*sql*/`
            SELECT COALESCE(MAX(message_index) + 1, 0) as next_index
            FROM conversations
            WHERE user_id = ? AND session_id = ? AND agent_id = ?
          `,
          args: [userId, sessionId, agentId]
        });

        const nextIndex = nextIndexResult.rows[0].next_index as number;
        const timestamp = Date.now();
        const content = Array.isArray(newMessage.content)
          ? JSON.stringify(newMessage.content)
          : JSON.stringify([{ text: newMessage.content }]);

        // Insert new message
        await tx.execute({
          sql: /*sql*/`
            INSERT INTO conversations (
              user_id, session_id, agent_id, message_index,
              role, content, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
          `,
          args: [
            userId,
            sessionId,
            agentId,
            nextIndex,
            newMessage.role,
            content,
            timestamp
          ]
        });

        // If maxHistorySize is set, cleanup old messages
        if (maxHistorySize !== undefined) {
          await tx.execute({
            sql: /*sql*/`
              DELETE FROM conversations
              WHERE user_id = ?
                AND session_id = ?
                AND agent_id = ?
                AND message_index <= (
                  SELECT MAX(message_index) - ?
                  FROM conversations
                  WHERE user_id = ?
                    AND session_id = ?
                    AND agent_id = ?
                )
            `,
            args: [
              userId, sessionId, agentId,
              maxHistorySize,
              userId, sessionId, agentId
            ]
          });
        }
      });

      // Return updated conversation
      return this.fetchChat(userId, sessionId, agentId, maxHistorySize);
    } catch (error) {
      Logger.logger.error("Error saving message:", error);
      throw error;
    }
  }

  async fetchChat(
    userId: string,
    sessionId: string,
    agentId: string,
    maxHistorySize?: number
  ): Promise<ConversationMessage[]> {
    try {
      const sql = /*sql*/`
        SELECT role, content, timestamp
        FROM conversations
        WHERE user_id = ? AND session_id = ? AND agent_id = ?
        ORDER BY message_index ${maxHistorySize ? 'DESC LIMIT ?' : 'ASC'}
      `;

      const args = maxHistorySize
        ? [userId, sessionId, agentId, maxHistorySize]
        : [userId, sessionId, agentId];

      const result = await this.client.execute({ sql, args });
      const messages = result.rows;

      // If we used LIMIT, we need to reverse the results to maintain chronological order
      if (maxHistorySize) messages.reverse();

      return messages.map(msg => ({
        role: msg.role as ParticipantRole,
        content: JSON.parse(msg.content as string),
      }));
    } catch (error) {
      Logger.logger.error("Error fetching chat:", error);
      throw error;
    }
  }

  async fetchAllChats(
    userId: string,
    sessionId: string
  ): Promise<ConversationMessage[]> {
    try {
      const result = await this.client.execute({
        sql: /*sql*/`
          SELECT role, content, timestamp, agent_id
          FROM conversations
          WHERE user_id = ? AND session_id = ?
          ORDER BY timestamp ASC
        `,
        args: [userId, sessionId]
      });

      const messages = result.rows;

      return messages.map(msg => ({
        role: msg.role as ParticipantRole,
        content: msg.role === ParticipantRole.ASSISTANT
          ? [{ text: `[${msg.agent_id}] ${JSON.parse(msg.content as string)[0]?.text || ''}` }]
          : JSON.parse(msg.content as string)
      }));
    } catch (error) {
      Logger.logger.error("Error fetching all chats:", error);
      throw error;
    }
  }

  async waitForInitialization() {
    if (this.client.closed) {
      throw new Error("Database connection closed");
    }
    await this.initialized;
  }

  close() {
    this.client.close();
  }
}
