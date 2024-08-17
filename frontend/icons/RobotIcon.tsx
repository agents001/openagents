import React from 'react';
import { FaRobot } from "react-icons/fa";

import RobotIconImage from './logo_rect.svg';

const RobotIcon = (props: React.HTMLProps<HTMLImageElement>) => {
  // return <img src={RobotIconImage.src} alt="Robot Icon" {...props} />;
  return <FaRobot className="ml-28" style={{ width: '30', height: '30' }}/>
};

export default RobotIcon;