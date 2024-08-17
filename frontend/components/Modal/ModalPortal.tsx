import ReactDOM from 'react-dom';

const ModalPortal: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const modalRoot = document.documentElement;
  return modalRoot ? ReactDOM.createPortal(children, modalRoot) : null;
};

export default ModalPortal;