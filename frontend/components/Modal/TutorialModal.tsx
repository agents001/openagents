import Modal from 'react-modal';
import ModalPortal from './ModalPortal';

const TutorialModal: React.FC<{ isOpen: boolean; onClose: () => void }> = ({ isOpen, onClose }) => {
  return (
    <ModalPortal>
      <Modal
        isOpen={isOpen}
        onRequestClose={onClose}
        contentLabel="新一轮对话说明"
        className="fixed inset-0 flex items-center justify-center"
        overlayClassName="fixed inset-0 bg-black bg-opacity-75 z-[100]"
      >
        <div className="bg-white rounded-lg shadow-lg p-6 max-w-xl w-full">
        <h2 className="text-xl font-semibold mb-4">如何开启新一轮的对话？</h2>
        <p className="mb-4">
          欢迎来到新的对话环节！在这里，您可以与我们的支持团队进行交流，获取帮助和解决问题。
          <br />
          请在下方输入您的问题或反馈，我们会尽快回复您。
        </p>
        <div className="flex flex justify-center">
          <button
            onClick={onClose}
            className="bg-blue-300 text-gray-700 rounded-md px-4 py-2 ml-2 hover:bg-gray-400"
          >
            我知道了
          </button>
        </div>
      </div>
      </Modal>
    </ModalPortal>
  );
};

export default TutorialModal;