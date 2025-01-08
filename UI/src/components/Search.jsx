import React, { useState, useContext } from "react";
import { useSearchParams } from "react-router-dom";
import { PlayerContext } from "../context/PlayerContext"


const SearchComponent = ({ setSearchHistory }) => {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [songs, setSongs] = useState([]);
  const {setIsLoadingSong,setCurrentTracks} = useContext(PlayerContext)

  const updateQueryParams = () => {
    const params = new URLSearchParams(searchParams);
    params.set("myArray", array.join(",")); // Convert array to a comma-separated string
    setSearchParams(params);
  };

  const handleSearch = async (e) => {
    if (e.key === "Enter" && query.trim()) {
      setLoading(true);

      const latestSearch = query;
      localStorage.setItem("latestSearch", latestSearch);
      setIsLoadingSong(true);
      const response = await fetch(
        "https://inference-app.cadentia.ai/get_emotion_based_song",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            text: latestSearch,
          }),
        }
      );
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const result = await response.json();
      const results = result["track_id"];
      setCurrentTracks(results)

      // Add to search history in reverse order
      setSearchHistory((prev) => [query, ...prev]);
      setQuery("");
    }
  };

  return (
    <div className="max-w-lg pl-3">
      {/* Search Input */}
      <div className="relative">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleSearch}
          className="w-full  rounded-none focus:outline-none bg-transparent"
          placeholder="Search..."
        />
      </div>
    </div>
  );
};

export default SearchComponent;
