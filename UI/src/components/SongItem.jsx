import { useContext } from "react";
import { PlayerContext } from "../context/PlayerContext";

const SpotifySong = ({ trackId }) => {
  if (!trackId) {
    return <p>Please provide a valid Spotify track ID.</p>;
  }
  return (
    <div>
      <iframe
        style={{ borderRadius: "10px" }}
        src={`https://open.spotify.com/embed/track/${trackId}?utm_source=generator`}
        width="100%"
        height="200"
        frameBorder="0"
        allowFullScreen
        allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
      ></iframe>
    </div>
  );
};

const SongItem = ({id}) => {
  const { playWithId } = useContext(PlayerContext);
  return (
    <div
      onClick={() => playWithId(id)}
      className="p-2 px-4 rounded cursor-pointer hover:bg-[#ffffff26]"
    >
      <SpotifySong trackId={id} />
    </div>
  );
};

export default SongItem;
