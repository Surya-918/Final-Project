import { useContext, useState } from "react";
import "./Main.css";
import { assets } from "../../assets/assets";
import { Context } from "../../context/Context";
import jsPDF from "jspdf";

const Main = () => {
  const {
    onSent,
    recentPrompt,
    showResult,
    loading,
    resultData,
    setInput,
    input,
    genre,
    setGenre,
    reload,
    modelWarning,
  } = useContext(Context);

  const [themes] = useState(["default", "sunrise", "forest"]);
  const [currentThemeIndex, setCurrentThemeIndex] = useState(0);

  const changeTheme = () => {
    const nextIndex = (currentThemeIndex + 1) % themes.length;
    setCurrentThemeIndex(nextIndex);
  };

  const handleSaveStory = () => {
    const doc = new jsPDF();
    const text = resultData;

    const pageWidth = doc.internal.pageSize.getWidth();
    const margin = 10;
    const maxLineWidth = pageWidth - margin * 2;

    const lines = doc.splitTextToSize(text, maxLineWidth);

    doc.text(lines, margin, margin);
    doc.save("story.pdf");
  };

  return (
    <div className={`main ${themes[currentThemeIndex]}`}>
      <div className="nav">
        <img
          src="https://www.svgrepo.com/show/387102/change.svg"
          alt="User Icon"
          onClick={changeTheme}
          style={{
            cursor: "pointer",
            filter:
              "invert(100%) sepia(0%) saturate(0%) hue-rotate(0deg) brightness(100%) contrast(100%)",
          }}
        />
      </div>
      <div className="main-container">
        {!showResult ? (
          <>
            <div className="greet">
              <p>
                <span>Hello, Poet</span>
              </p>
              <p>Let&apos;s create innovative stories</p>
            </div>
            <div className="cards">
              <div
                className="card card1"
                onClick={() => {
                  setGenre("action");
                  setInput(
                    "A sudden campus emergency forces students to secure the research wing."
                  );
                }}
              >
                <p>Action Stories</p>
              </div>
              <div
                className="card card2"
                onClick={() => {
                  setGenre("drama");
                  setInput(
                    "Two roommates stop speaking after a scholarship secret comes out."
                  );
                }}
              >
                <p>Drama Stories</p>
              </div>
              <div
                className="card card3"
                onClick={() => {
                  setGenre("action");
                  setInput(
                    "A robotics team has minutes to get everyone out of a smoking arena."
                  );
                }}
              >
                <p>Action Stories</p>
              </div>
              <div
                className="card card4"
                onClick={() => {
                  setGenre("drama");
                  setInput(
                    "A student finds an unsent letter in an old library book that changes everything."
                  );
                }}
              >
                <p>Drama Stories</p>
              </div>
            </div>
          </>
        ) : (
          <div className="result">
            <div className="result-title">
              <img
                src="https://www.svgrepo.com/show/51941/male-user.svg"
                alt=""
                style={{
                  filter:
                    "invert(100%) sepia(0%) saturate(0%) hue-rotate(0deg) brightness(100%) contrast(100%)",
                }}
              />
              <p>{recentPrompt}</p>
            </div>
            <div className="result-data">
              <img src={assets.narrative} alt="Gemini Icon" />
              {loading ? (
                <div className="loader">
                  <hr />
                  <hr />
                  <hr />
                </div>
              ) : (
                <p>{resultData}</p>
              )}
            </div>
          </div>
        )}
        <div className="main-bottom">
          <div className="search-box">
            <input
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && input.trim() && !loading) {
                  onSent(input);
                }
              }}
              value={input}
              type="text"
              placeholder="Enter a story idea here"
            />
            <select
              className="genre-select"
              aria-label="Story genre"
              value={genre}
              onChange={(event) => setGenre(event.target.value)}
            >
              <option value="drama">Drama</option>
              <option value="action">Action</option>
            </select>
            <div>
              <img onClick={handleSaveStory} src={assets.save} alt="" />
              <img
                onClick={() => reload(recentPrompt)}
                src={assets.Regenerate}
                alt=""
              />
              {input ? (
                <img
                  onClick={() => onSent(input)}
                  src={assets.send_icon}
                  alt="Send"
                />
              ) : null}
            </div>
          </div>
          {modelWarning ? (
            <p className="model-warning">{modelWarning}</p>
          ) : null}
        </div>
      </div>
    </div>
  );
};

export default Main;
