import { createContext, useState, useEffect } from "react";
import PropTypes from "prop-types";
import axios from "axios";

export const Context = createContext();
const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const DEMO_MODE = import.meta.env.VITE_DEMO_MODE === "true";
const createConversationId = () =>
  globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`;

const getDemoStory = (prompt) => {
  const idea = prompt || "an unexpected adventure began";
  const templates = [
    `At midnight, ${idea}. A strange map appeared under the moonlight, leading to a place no one had ever seen. The journey was filled with impossible choices, but by sunrise, the hero had found the courage to change the ending.`,
    `Everyone remembered the day when ${idea}. At first it sounded like a rumor, until the old clock tower opened a hidden passage. Inside was a secret that connected the past, present, and future in one unforgettable adventure.`,
    `The rain had just stopped when ${idea}. A silver light crossed the sky and revealed a forgotten message: "Your story begins now." With every step forward, danger grew closer, but so did an unexpected friendship.`,
    `Long ago, ${idea}. The village thought the mystery was impossible to solve, but one brave traveler followed a trail of glowing stars. The discovery at the end became a legend told for generations.`,
  ];
  const index = [...idea].reduce((total, character) => total + character.charCodeAt(0), 0) % templates.length;
  return templates[index];
};

const ContextProvider = ({ children }) => {
  const [input, setInput] = useState("");
  const [genre, setGenre] = useState(() => {
    const saved = localStorage.getItem("storyGenre");
    return saved === "action" || saved === "drama" ? saved : "drama";
  });
  const [recentPrompt, setRecentPrompt] = useState("");
  const [conversationId, setConversationId] = useState(
    () => localStorage.getItem("conversationId") || createConversationId()
  );
  const [storyId, setStoryId] = useState(
    () => localStorage.getItem("storyId") || ""
  );
  const [prevPrompts, setPrevPrompts] = useState([]);
  const [showResult, setShowResult] = useState(false);
  const [loading, setLoading] = useState(false);
  const [resultData, setResultData] = useState("");
  const [modelWarning, setModelWarning] = useState("");

  useEffect(() => {
    try {
      const savedPrompts = JSON.parse(localStorage.getItem("prompts") || "[]");
      if (Array.isArray(savedPrompts)) {
        setPrevPrompts(savedPrompts);
      }
    } catch (error) {
      console.warn("Could not read saved prompts:", error);
      localStorage.removeItem("prompts");
      setPrevPrompts([]);
    }
  }, []);

  useEffect(() => {
    axios
      .get(`${API_URL}/health`, { timeout: 8000 })
      .then((res) => {
        if (res.data?.warning) {
          setModelWarning(res.data.warning);
        }
      })
      .catch(() => {
        setModelWarning(
          "Story API is not running on port 8000. Start it with npm run backend."
        );
      });
  }, []);

  useEffect(() => {
    localStorage.setItem("conversationId", conversationId);
  }, [conversationId]);

  useEffect(() => {
    localStorage.setItem("storyGenre", genre);
  }, [genre]);

  useEffect(() => {
    localStorage.setItem("storyId", storyId);
  }, [storyId]);

  const saveToLocalStorage = (prompt, result, activeStoryId) => {
    try {
      const rawPrompts = localStorage.getItem("prompts");
      const savedPrompts = rawPrompts ? JSON.parse(rawPrompts) : [];
      const safePrompts = Array.isArray(savedPrompts) ? savedPrompts : [];
      const newPrompt = {
        prompt,
        result,
        conversationId,
        storyId: activeStoryId,
        genre,
      };
      const dedupedPrompts = safePrompts.filter(
        (entry) => !(entry.storyId && entry.storyId === activeStoryId)
      );
      dedupedPrompts.unshift(newPrompt);
      const trimmedPrompts = dedupedPrompts.slice(0, 12);
      localStorage.setItem("prompts", JSON.stringify(trimmedPrompts));
      setPrevPrompts(trimmedPrompts);
    } catch (error) {
      console.warn("Could not save prompt history:", error);
    }
  };

  const newChat = () => {
    setLoading(false);
    setShowResult(false);
    setInput("");
    setResultData("");
    setConversationId(createConversationId());
    setStoryId("");
  };

  const selectGenre = (nextGenre) => {
    if (nextGenre !== genre) {
      setStoryId(createConversationId());
    }
    setGenre(nextGenre);
  };

  const restorePrompt = (savedPrompt) => {
    setConversationId(savedPrompt.conversationId || createConversationId());
    setStoryId(savedPrompt.storyId || "");
    setRecentPrompt(savedPrompt.prompt);
    setResultData(savedPrompt.result);
    setGenre(savedPrompt.genre === "action" ? "action" : "drama");
    setShowResult(true);
    setLoading(false);
    setInput("");
  };

  const onSent = async (prompt) => {
    setResultData("");
    setLoading(true);
    setShowResult(true);
    if (!prompt) {
      prompt = input;
    }
    setRecentPrompt(prompt);
    try {
      const res = await axios.post(
        `${API_URL}/generate`,
        {
          prompt,
          conversation_id: conversationId,
          story_id: storyId,
          genre,
          max_length: 180,
        },
        { timeout: 120000 }
      );
      const response = res.data.generated_text;
      const activeStoryId = res.data.story_id || "";
      setStoryId(activeStoryId);
      saveToLocalStorage(prompt, response, activeStoryId);
      setResultData(response);
    } catch (error) {
      console.error("Error generating story:", error);
      const detail = error.response?.data?.detail;
      setResultData(
        DEMO_MODE
          ? getDemoStory(prompt)
          : detail
            ? `Sorry, story generation failed: ${detail}`
            : "Sorry, there was an error generating the story. Is the backend running on port 8000?"
      );
    } finally {
      setLoading(false);
      setInput("");
    }
  };

  const reload = async (prompt) => {
    setResultData("");
    setLoading(true);
    setShowResult(true);
    try {
      const res = await axios.post(
        `${API_URL}/generate`,
        {
          prompt,
          conversation_id: conversationId,
          story_id: storyId,
          genre,
          max_length: 180,
        },
        { timeout: 120000 }
      );
      const response = res.data.generated_text;
      const activeStoryId = res.data.story_id || "";
      setStoryId(activeStoryId);
      saveToLocalStorage(prompt, response, activeStoryId);
      setResultData(response);
    } catch (error) {
      console.error("Error generating story:", error);
      const detail = error.response?.data?.detail;
      setResultData(
        DEMO_MODE
          ? getDemoStory(prompt)
          : detail
            ? `Sorry, story generation failed: ${detail}`
            : "Sorry, there was an error generating the story. Is the backend running on port 8000?"
      );
    } finally {
      setLoading(false);
      setInput("");
    }
  };

  const contextValue = {
    prevPrompts,
    setPrevPrompts,
    onSent,
    restorePrompt,
    setRecentPrompt,
    recentPrompt,
    showResult,
    loading,
    resultData,
    input,
    setInput,
    genre,
    setGenre: selectGenre,
    newChat,
    reload,
    modelWarning,
  };

  return <Context.Provider value={contextValue}>{children}</Context.Provider>;
};

ContextProvider.propTypes = {
  children: PropTypes.node.isRequired,
};

export default ContextProvider;
