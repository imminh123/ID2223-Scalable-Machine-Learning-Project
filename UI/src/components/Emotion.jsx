import React, { useState, useEffect } from "react";
import { use } from "react";

const EmotionFeedback = ({isSubmitted, setIsSubmitted}) => {
  const emotions = [
    "Happiness",
    "Contentment",
    "Confidence",
    "Neutral",
    "Sadness",
    "Anger",
    "Fear",
    "Surprise",
    "Disgust",
    "Love",
    "Excitement",
    "Anticipation",
    "Nostalgia",
    "Confusion",
    "Frustration",
    "Longing",
    "Optimism",
  ];

  const [selectedEmotions, setSelectedEmotions] = useState([]);
  
  useEffect(() => {
    setSelectedEmotions([]);
  }, [isSubmitted]);

  const toggleEmotion = (emotion) => {
    setSelectedEmotions((prev) =>
      prev.includes(emotion)
        ? prev.filter((e) => e !== emotion)
        : [...prev, emotion]
    );
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const latestSearch = localStorage.getItem("latestSearch");
      const response = await fetch("https://inference-app.cadentia.ai/feedback", {
        method: "POST",
        headers: {
          "Content-Type": "application/json", 
        },
        body: JSON.stringify({
          text: latestSearch,
          emotions: selectedEmotions,
        }), 
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const result = await response.json();
      setIsSubmitted(true);
    } catch (err) {
      console.error(err);
    }
  };


  return (
    <div className="max-w-md p-4 shadow-md bg-gray-900 text-white">
      {!isSubmitted ? (
        <>
          <h2 className="text-lg font-semibold  mb-4">
            Did we get the song you were looking for?
          </h2>
          <p className="text-gray-100 mb-3">If not, let us know how do you feel right now</p>
          {/* Emotion Chips */}
          <div className="flex flex-wrap gap-2 mb-4">
            {emotions.map((emotion) => (
              <button
                key={emotion}
                onClick={() => toggleEmotion(emotion)}
                className={`px-3 py-1 rounded-full text-sm font-medium border ${
                  selectedEmotions.includes(emotion)
                    ? "bg-green-700 text-white border-green-700 hover:bg-green-500"
                    : "bg-gray-200 text-gray-700 border-gray-300 hover:bg-green-100"
                } `}
              >
                {emotion}
              </button>
            ))}
          </div>
          {/* Submit Button */}
          <button
            onClick={handleSubmit}
            disabled={selectedEmotions.length === 0}
            className="px-4 py-2 rounded-full bg-green-700 text-white text-sm font-medium hover:bg-green-800 disabled:bg-gray-300"
          >
            Submit
          </button>
        </>
      ) : (
        <div className="text-center">
          <h3 className="text-lg font-semibold  mb-2">Thank You!</h3>
          <p className="text-gray-100">
            We appreciate your feedback. We'll strive to recommend better songs next time!
          </p>
        </div>
      )}
    </div>
  );
};

export default EmotionFeedback;