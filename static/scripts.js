const X_CLASS = 'x';
const O_CLASS = 'o';
const WINNING_COMBINATIONS = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6]
];

const cellElements = document.querySelectorAll('[data-cell]');
const board = document.getElementById('board');
const winningMessageElement = document.getElementById('message');
const winningMessageTextElement = document.getElementById('winningMessageText');
const restartButton = document.getElementById('restartButton');
const playerXScoreElement = document.getElementById('playerXScore');
const playerOScoreElement = document.getElementById('playerOScore');
let oTurn;
let playerXScore = 0;
let playerOScore = 0;

startGame();

restartButton.addEventListener('click', startGame);

function startGame() {
    oTurn = false;
    cellElements.forEach(cell => {
        cell.classList.remove(X_CLASS);
        cell.classList.remove(O_CLASS);
        cell.querySelector('.mark').innerText = '';
        cell.classList.remove('winning');
        cell.removeEventListener('click', handleClick);
        cell.addEventListener('click', handleClick, { once: true });
    });
    setBoardHoverClass();
    winningMessageElement.classList.remove('show');
}

function handleClick(e) {
    const cell = e.target;
    const mark = oTurn ? 'O' : 'X';
    placeMark(cell, mark);
    if (checkWin(mark)) {
        endGame(false, mark);
    } else if (isDraw()) {
        endGame(true);
    } else {
        swapTurns();
        setBoardHoverClass();
    }
}

function endGame(draw, mark) {
    if (draw) {
        winningMessageTextElement.innerText = 'Draw!';
    } else {
        winningMessageTextElement.innerText = `${mark}'s Wins!`;
        mark === 'X' ? playerXScore++ : playerOScore++;
        updateScore();
    }
    winningMessageElement.classList.add('show');
}

function placeMark(cell, mark) {
    const markElement = cell.querySelector('.mark');
    markElement.innerText = mark;
}

function swapTurns() {
    oTurn = !oTurn;
}

function setBoardHoverClass() {
    board.classList.remove(X_CLASS);
    board.classList.remove(O_CLASS);
    if (oTurn) {
        board.classList.add(O_CLASS);
    } else {
        board.classList.add(X_CLASS);
    }
}

function checkWin(mark) {
    return WINNING_COMBINATIONS.some(combination => {
        return combination.every(index => {
            return cellElements[index].querySelector('.mark').innerText === mark;
        });
    });
}

function isDraw() {
    return [...cellElements].every(cell => {
        return cell.querySelector('.mark').innerText !== '';
    });
}

function updateScore() {
    playerXScoreElement.innerText = playerXScore;
    playerOScoreElement.innerText = playerOScore;
}
