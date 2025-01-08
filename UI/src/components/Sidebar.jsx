import { useNavigate } from "react-router-dom";
import { useState } from "react";
import { assets } from "../assets/assets";
import Search from "./Search";

const Sidebar = () => {
  const navigate = useNavigate();
  const [searchHistory, setSearchHistory] = useState([]);

  const removeHistoryItem = (index) => {
    setSearchHistory((prev) => prev.filter((_, i) => i !== index));
  };

  return (
    <div className="w-[25%] h-full p-2 flex-col gap-2 text-white lg:flex">
      <div className="bg-[#121212] h-[9%] rounded flex flex-col justify-around">
        {/* <div onClick={()=>navigate('/')} className="flex items-center gap-3 pl-8 cursor-pointer">
          <img className="w-6" src={assets.home_icon} alt="" />
          <p className="font-b old">Home</p>
        </div> */}
        <div className="flex items-center pl-3 cursor-pointer">
          <img className="w-6" src={assets.search_icon} alt="" />
          <Search setSearchHistory={setSearchHistory} />
        </div>

      </div>

      {/* Search History */}
      <div className="p-1">
        <ul className="space-y-2">
          {searchHistory.map((item, index) => (
            <li
              key={index}
              className="flex items-center justify-between px-1 py-1 rounded-lg shadow"
            >
              <span>{item}</span>
              <button
               onClick={() => removeHistoryItem(index)}
              className="text-gray-300 hover:underline">x</button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default Sidebar;
